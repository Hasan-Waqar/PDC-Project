#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <limits>
#include <algorithm>
#include <mpi.h>
#include <metis.h>
#include <cstdlib>
#include <ctime>
#include <omp.h> // Added OpenMP header

using namespace std;

// Edge structure
struct Edge {
    int u, v;
    double w;
};

// Read graph
void readGraph(const string& filename, int& numVertices, long long& numEdges,
               unordered_map<int, int>& idMap, vector<pair<int, int>>& edgeList) {
    ifstream file(filename.c_str());
    if (!file.is_open()) {
        cerr << "Error: Could not open " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    string line;
    long long edgesRead = 0;
    unordered_set<int> vertices;
    edgeList.clear();
    int expectedNodes = 0;
    long long expectedEdges = 0;

    // Read header first
    while (getline(file, line)) {
        if (line.empty() || line[0] != '#') break;
        if (line.find("Nodes:") != string::npos) {
            sscanf(line.c_str(), "# Nodes: %d Edges: %lld", &expectedNodes, &expectedEdges);
        }
    }

    // Pre-allocate vectors if we know the size
    if (expectedEdges > 0) {
        edgeList.reserve(expectedEdges);
    }

    // Read all lines into memory first for parallel processing
    vector<string> lines;
    lines.push_back(line); // Add the first non-header line
    while (getline(file, line)) {
        lines.push_back(line);
    }
    file.close();

    // Process edges in parallel using OpenMP
    #pragma omp parallel
    {
        unordered_set<int> local_vertices;
        vector<pair<int, int>> local_edgeList;
        local_edgeList.reserve(expectedEdges / omp_get_num_threads() + 1);

        #pragma omp for schedule(dynamic, 10000)
        for (size_t i = 0; i < lines.size(); i++) {
            const string& line = lines[i];
            if (line.empty() || line[0] == '#') continue;
            
            stringstream ss(line);
            int u, v;
            if (ss >> u >> v) {
                local_vertices.insert(u);
                local_vertices.insert(v);
                local_edgeList.push_back({u, v});
            }
        }

        // Merge local results
        #pragma omp critical
        {
            vertices.insert(local_vertices.begin(), local_vertices.end());
            edgeList.insert(edgeList.end(), local_edgeList.begin(), local_edgeList.end());
            edgesRead += local_edgeList.size();
        }
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0 && edgesRead % 1000000 == 0) {
        cout << "Scanned " << edgesRead << " edges..." << endl;
    }

    if (vertices.empty()) {
        cerr << "Error: No valid vertices found in " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    numVertices = vertices.size();
    int newId = 0;
    idMap.reserve(numVertices);
    for (auto v : vertices) {
        idMap[v] = newId++;
    }

    // Map edge IDs in parallel with better scheduling
    #pragma omp parallel for schedule(dynamic, 10000)
    for (size_t i = 0; i < edgeList.size(); i++) {
        edgeList[i].first = idMap[edgeList[i].first];
        edgeList[i].second = idMap[edgeList[i].second];
    }

    numEdges = edgesRead;

    if (rank == 0) {
        cout << "Graph Statistics:" << endl;
        cout << "----------------" << endl;
        cout << "Vertices: " << numVertices << endl;
        cout << "Edges: " << edgesRead << endl;
    }
}

// Convert to CSR for METIS
void convertToCSR(const vector<pair<int, int>>& edgeList, int numVertices,
                  vector<idx_t>& xadj, vector<idx_t>& adjncy) {
    vector<vector<int>> adj(numVertices);
    
    // Build adjacency lists in parallel
    #pragma omp parallel
    {
        vector<vector<int>> local_adj(numVertices);
        
        #pragma omp for schedule(dynamic, 10000)
        for (size_t i = 0; i < edgeList.size(); i++) {
            const auto& edge = edgeList[i];
            local_adj[edge.first].push_back(edge.second);
            local_adj[edge.second].push_back(edge.first); // METIS requires undirected
        }
        
        // Merge local adjacency lists
        #pragma omp critical
        {
            for (int u = 0; u < numVertices; ++u) {
                adj[u].insert(adj[u].end(), local_adj[u].begin(), local_adj[u].end());
            }
        }
    }

    // Sort adjacency lists to remove duplicates
    #pragma omp parallel for schedule(dynamic, 1000)
    for (int u = 0; u < numVertices; ++u) {
        sort(adj[u].begin(), adj[u].end());
        auto it = unique(adj[u].begin(), adj[u].end());
        adj[u].resize(it - adj[u].begin());
    }

    // Compute CSR format
    xadj.resize(numVertices + 1);
    xadj[0] = 0;
    
    // Calculate offsets
    for (int u = 0; u < numVertices; ++u) {
        xadj[u + 1] = xadj[u] + adj[u].size();
    }
    
    // Allocate and fill adjacency array
    adjncy.resize(xadj[numVertices]);
    
    #pragma omp parallel for schedule(dynamic, 1000)
    for (int u = 0; u < numVertices; ++u) {
        for (size_t j = 0; j < adj[u].size(); ++j) {
            adjncy[xadj[u] + j] = adj[u][j];
        }
    }
}

// Partition graph with METIS
void partitionGraph(int numVertices, vector<idx_t>& xadj, vector<idx_t>& adjncy,
                   int nparts, vector<idx_t>& part) {
    idx_t nvtxs = numVertices;
    idx_t ncon = 1;
    idx_t objval;
    part.resize(numVertices);

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
    options[METIS_OPTION_NUMBERING] = 0;

    int ret = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(),
                                  NULL, NULL, NULL, &nparts, NULL, NULL,
                                  options, &objval, part.data());
    if (ret != METIS_OK) {
        cerr << "METIS partitioning failed with error code " << ret << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        cout << "METIS partitioning completed. Edge-cut: " << objval << endl;
    }
}

// Distribute subgraphs
void distributeSubgraphs(const vector<pair<int, int>>& edgeList, const vector<idx_t>& part,
                        int rank, int size, int numVertices,
                        vector<vector<pair<int, double>>>& localAdj,
                        vector<int>& localVertices,
                        unordered_set<int>& ghostVertices) {
    localAdj.assign(numVertices, {});
    localVertices.clear();
    ghostVertices.clear();

    // Identify local vertices in parallel
    vector<bool> isLocalVertex(numVertices, false);
    
    #pragma omp parallel for schedule(static)
    for (int v = 0; v < numVertices; ++v) {
        if (part[v] == rank) {
            isLocalVertex[v] = true;
        }
    }
    
    for (int v = 0; v < numVertices; ++v) {
        if (isLocalVertex[v]) {
            localVertices.push_back(v);
        }
    }

    // Process edges and identify ghost vertices in parallel
    vector<unordered_set<int>> thread_ghost_vertices;
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        
        // Initialize thread-local ghost vertex sets
        #pragma omp single
        {
            thread_ghost_vertices.resize(num_threads);
        }
        
        #pragma omp for schedule(dynamic, 10000)
        for (size_t i = 0; i < edgeList.size(); i++) {
            const auto& edge = edgeList[i];
            int u = edge.first, v = edge.second;
            double w = 1.0; // Default weight
            if (part[u] == rank) {
                #pragma omp critical
                {
                    localAdj[u].push_back({v, w});
                }
                if (part[v] != rank) {
                    thread_ghost_vertices[thread_id].insert(v);
                }
            }
        }
    }
    
    // Merge ghost vertices
    for (const auto& thread_ghosts : thread_ghost_vertices) {
        ghostVertices.insert(thread_ghosts.begin(), thread_ghosts.end());
    }
}

// Initialize updates (100,000 random operations)
void initializeUpdates(vector<pair<Edge, bool>>& changes, int numVertices,
                       const vector<pair<int, int>>& edgeList) {
    int NUM_CHANGES = 100;
    changes.clear();
    changes.reserve(NUM_CHANGES);
    unordered_set<long long> existingEdges;
    
    // Build existing edges map in parallel
    #pragma omp parallel
    {
        unordered_set<long long> thread_existingEdges;
        
        #pragma omp for schedule(dynamic, 10000)
        for (size_t i = 0; i < edgeList.size(); i++) {
            const auto& edge = edgeList[i];
            long long edgeKey = static_cast<long long>(edge.first) * numVertices + edge.second;
            thread_existingEdges.insert(edgeKey);
        }
        
        #pragma omp critical
        {
            existingEdges.insert(thread_existingEdges.begin(), thread_existingEdges.end());
        }
    }

    int targetChanges = 100;
    int additions = 0, deletions = 0, weightUpdates = 0;
    vector<int> vertices(numVertices);
    for (int i = 0; i < numVertices; ++i) {
        vertices[i] = i;
    }

    srand(time(0));
    for (int i = 0; i < targetChanges; ++i) {
        int op = rand() % 3;
        int u, v;
        double w = (rand() % 95 + 5) / 10.0; // Random weight 0.5 to 10.0
        long long edgeKey;

        if (op == 0 && additions < NUM_CHANGES/3) { // Addition
            do {
                u = vertices[rand() % vertices.size()];
                v = vertices[rand() % vertices.size()];
                if (u != v) {
                    edgeKey = static_cast<long long>(u) * numVertices + v;
                }
            } while (u == v || existingEdges.find(edgeKey) != existingEdges.end());
            changes.push_back({Edge{u, v, w}, true});
            existingEdges.insert(edgeKey);
            additions++;
        } else if (op == 1 && deletions < NUM_CHANGES/3 && !edgeList.empty()) { // Deletion
            int idx = rand() % edgeList.size();
            u = edgeList[idx].first;
            v = edgeList[idx].second;
            edgeKey = static_cast<long long>(u) * numVertices + v;
            if (existingEdges.find(edgeKey) != existingEdges.end()) {
                changes.push_back({Edge{u, v, 1.0}, false});
                existingEdges.erase(edgeKey);
                deletions++;
            }
        } else if (weightUpdates < NUM_CHANGES/3 && !edgeList.empty()) { // Weight update
            int idx = rand() % edgeList.size();
            u = edgeList[idx].first;
            v = edgeList[idx].second;
            edgeKey = static_cast<long long>(u) * numVertices + v;
            if (existingEdges.find(edgeKey) != existingEdges.end()) {
                changes.push_back({Edge{u, v, 1.0}, false}); // Delete
                existingEdges.erase(edgeKey);
                changes.push_back({Edge{u, v, w}, true});   // Add with new weight
                existingEdges.insert(edgeKey);
                weightUpdates += 2;
                i++;
            }
        } else {
            do {
                u = vertices[rand() % vertices.size()];
                v = vertices[rand() % vertices.size()];
                if (u != v) {
                    edgeKey = static_cast<long long>(u) * numVertices + v;
                }
            } while (u == v || existingEdges.find(edgeKey) != existingEdges.end());
            changes.push_back({Edge{u, v, w}, true});
            existingEdges.insert(edgeKey);
            additions++;
        }
    }

    while (changes.size() < NUM_CHANGES && additions < NUM_CHANGES/3) {
        int u, v;
        long long edgeKey;
        double w = (rand() % 95 + 5) / 10.0;
        do {
            u = vertices[rand() % vertices.size()];
            v = vertices[rand() % vertices.size()];
            if (u != v) {
                edgeKey = static_cast<long long>(u) * numVertices + v;
            }
        } while (u == v || existingEdges.find(edgeKey) != existingEdges.end());
        changes.push_back({Edge{u, v, w}, true});
        existingEdges.insert(edgeKey);
        additions++;
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        cout << "Generated " << changes.size() << " changes: " 
             << additions << " additions, " << deletions << " deletions, " 
             << weightUpdates << " weight updates" << endl;
    }
}

// Check if edge is in SSSP tree
bool isEdgeInSSSPTree(const vector<double>& dist, const vector<int>& parent, int u, int v) {
    return (parent[v] == u && dist[v] < numeric_limits<double>::infinity());
}

// Lock for priority queue operations
omp_lock_t pq_lock;

// Parallel incremental SSSP update
void updateSSSP(const Edge& e, bool isInsertion,
                vector<vector<pair<int, double>>>& localAdj,
                const vector<int>& localVertices,
                unordered_set<int>& ghostVertices,
                const vector<idx_t>& part, int rank, int size, int numVertices,
                vector<double>& dist, vector<int>& parent) {
    // Initialize the OpenMP lock
    omp_lock_t pq_lock;
    omp_init_lock(&pq_lock);
    
    // Resize dist and parent if necessary
    int maxVertex = max(e.u, e.v) + 1;
    if (maxVertex > static_cast<int>(dist.size())) {
        dist.resize(maxVertex, numeric_limits<double>::infinity());
        parent.resize(maxVertex, -1);
    }

    // Identify affected vertex
    int x, y;
    if (dist[e.u] > dist[e.v]) {
        x = e.u;
        y = e.v;
    } else {
        x = e.v;
        y = e.u;
    }

    using P = pair<double, int>;
    priority_queue<P, vector<P>, greater<P>> pq;
    vector<bool> affected(numVertices, false);

    bool isLocal = part[e.u] == rank || part[e.v] == rank;
    if (isLocal) {
        if (isInsertion) {
            if (part[e.u] == rank) {
                #pragma omp critical
                {
                    localAdj[e.u].push_back({e.v, e.w});
                    if (part[e.v] != rank) ghostVertices.insert(e.v);
                }
            }
            double newDist = dist[y] + e.w;
            if (dist[x] > newDist && part[x] == rank) {
                dist[x] = newDist;
                parent[x] = y;
                affected[x] = true;
                omp_set_lock(&pq_lock);
                pq.push({dist[x], x});
                omp_unset_lock(&pq_lock);
            }
        } else {
            if (part[e.u] == rank && isEdgeInSSSPTree(dist, parent, e.u, e.v)) {
                if (parent[x] == y) {
                    dist[x] = numeric_limits<double>::infinity();
                    parent[x] = -1;
                    affected[x] = true;
                    omp_set_lock(&pq_lock);
                    pq.push({dist[x], x});
                    omp_unset_lock(&pq_lock);
                }
            }
            if (part[e.u] == rank) {
                #pragma omp critical
                {
                    auto& adj = localAdj[e.u];
                    adj.erase(remove_if(adj.begin(), adj.end(),
                                      [v = e.v](const auto& p) { return p.first == v; }),
                             adj.end());
                }
            }
        }
    }

    // Pre-allocate buffers for better performance
    vector<vector<pair<int, double>>> sendBuffers(size);
    vector<vector<pair<int, double>>> recvBuffers(size);
    for (int p = 0; p < size; ++p) {
        sendBuffers[p].reserve(numVertices / size);
        recvBuffers[p].reserve(numVertices / size);
    }

    bool globalChanges = true;
    while (globalChanges) {
        globalChanges = false;

        // Process priority queue in parallel
        #pragma omp parallel
        {
            vector<pair<double, int>> thread_updates;
            thread_updates.reserve(1000); // Pre-allocate for better performance

            while (true) {
                P current;
                bool has_item = false;
                
                omp_set_lock(&pq_lock);
                if (!pq.empty()) {
                    current = pq.top();
                    pq.pop();
                    has_item = true;
                }
                omp_unset_lock(&pq_lock);
                
                if (!has_item) break;
                
                double d = current.first;
                int u = current.second;
                
                if (!affected[u]) continue;
                affected[u] = false;

                // Process adjacency list in parallel
                #pragma omp for reduction(||:globalChanges) nowait
                for (size_t i = 0; i < localAdj[u].size(); i++) {
                    int v = localAdj[u][i].first;
                    double w = localAdj[u][i].second;
                    double newDist = (dist[u] == numeric_limits<double>::infinity())
                                  ? numeric_limits<double>::infinity()
                                  : dist[u] + w;
                    
                    bool local_update = false;
                    #pragma omp critical
                    {
                        if (newDist < dist[v]) {
                            dist[v] = newDist;
                            parent[v] = u;
                            local_update = true;
                        }
                    }
                    
                    if (local_update && (part[v] == rank || ghostVertices.count(v))) {
                        thread_updates.push_back({dist[v], v});
                        globalChanges = true;
                    }
                }
            }
            
            // Add thread-local updates to priority queue
            if (!thread_updates.empty()) {
                omp_set_lock(&pq_lock);
                for (const auto& update : thread_updates) {
                    pq.push(update);
                    affected[update.second] = true;
                }
                omp_unset_lock(&pq_lock);
            }
        }

        // Prepare send buffers in parallel
        #pragma omp parallel
        {
            vector<vector<pair<int, double>>> thread_sendBuffers(size);
            for (int p = 0; p < size; ++p) {
                thread_sendBuffers[p].reserve(numVertices / (size * omp_get_num_threads()));
            }
            
            #pragma omp for nowait
            for (int v = 0; v < numVertices; ++v) {
                if (affected[v] && ghostVertices.count(v)) {
                    int targetRank = part[v];
                    thread_sendBuffers[targetRank].push_back({v, dist[v]});
                }
            }
            
            // Combine thread-local buffers
            #pragma omp critical
            {
                for (int p = 0; p < size; ++p) {
                    sendBuffers[p].insert(sendBuffers[p].end(), 
                                         thread_sendBuffers[p].begin(), 
                                         thread_sendBuffers[p].end());
                }
            }
        }

        // Optimize communication by using non-blocking sends
        vector<MPI_Request> requests;
        requests.reserve(2 * (size - 1));
        
        for (int p = 0; p < size; ++p) {
            if (p == rank) continue;
            int count = sendBuffers[p].size();
            MPI_Request req;
            MPI_Isend(&count, 1, MPI_INT, p, 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
            if (count > 0) {
                MPI_Isend(sendBuffers[p].data(), count * sizeof(pair<int, double>),
                          MPI_BYTE, p, 1, MPI_COMM_WORLD, &req);
                requests.push_back(req);
            }
        }

        // Process received updates in parallel
        for (int p = 0; p < size; ++p) {
            if (p == rank) continue;
            int count;
            MPI_Recv(&count, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (count > 0) {
                recvBuffers[p].resize(count);
                MPI_Recv(recvBuffers[p].data(), count * sizeof(pair<int, double>),
                         MPI_BYTE, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                #pragma omp parallel
                {
                    vector<pair<double, int>> thread_updates;
                    thread_updates.reserve(count / omp_get_num_threads() + 1);
                    
                    #pragma omp for reduction(||:globalChanges) nowait
                    for (int i = 0; i < count; i++) {
                        int v = recvBuffers[p][i].first;
                        double newDist = recvBuffers[p][i].second;
                        
                        bool local_update = false;
                        #pragma omp critical
                        {
                            if (newDist < dist[v]) {
                                dist[v] = newDist;
                                parent[v] = -1;
                                local_update = true;
                            }
                        }
                        
                        if (local_update && part[v] == rank) {
                            thread_updates.push_back({newDist, v});
                            globalChanges = true;
                        }
                    }
                    
                    if (!thread_updates.empty()) {
                        omp_set_lock(&pq_lock);
                        for (const auto& update : thread_updates) {
                            pq.push(update);
                            affected[update.second] = true;
                        }
                        omp_unset_lock(&pq_lock);
                    }
                }
            }
        }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

        // Clear send buffers for next iteration
        for (int p = 0; p < size; ++p) {
            sendBuffers[p].clear();
        }

        int localChanges = globalChanges ? 1 : 0;
        int globalFlag;
        MPI_Allreduce(&localChanges, &globalFlag, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        globalChanges = globalFlag > 0;
    }
    
    // Destroy the lock
    omp_destroy_lock(&pq_lock);
}

// Compute initial SSSP (Dijkstra's)
void computeInitialSSSP(const vector<vector<pair<int, double>>>& localAdj,
                       const vector<int>& localVertices,
                       const unordered_set<int>& ghostVertices,
                       const vector<idx_t>& part, int rank, int size, int numVertices,
                       vector<double>& dist, vector<int>& parent, int source) {
    const double INF = numeric_limits<double>::infinity();
    dist.assign(numVertices, INF);
    parent.assign(numVertices, -1);

    if (part[source] == rank) {
        dist[source] = 0;
        parent[source] = -1;
    }

    using P = pair<double, int>;
    priority_queue<P, vector<P>, greater<P>> pq;
    vector<bool> visited(numVertices, false);
    if (part[source] == rank) {
        pq.push({0, source});
    }

    // Pre-allocate buffers for better performance
    vector<vector<pair<int, double>>> sendBuffers(size);
    vector<vector<pair<int, double>>> recvBuffers(size);
    for (int p = 0; p < size; ++p) {
        sendBuffers[p].reserve(numVertices / size);
        recvBuffers[p].reserve(numVertices / size);
    }

    bool globalChanges = true;
    while (globalChanges) {
        globalChanges = false;

        // Process priority queue in parallel
        #pragma omp parallel
        {
            vector<pair<double, int>> thread_updates;
            thread_updates.reserve(1000); // Pre-allocate for better performance

            while (true) {
                P current;
                bool has_item = false;
                
                #pragma omp critical
                {
                    if (!pq.empty()) {
                        current = pq.top();
                        pq.pop();
                        has_item = true;
                    }
                }
                
                if (!has_item) break;
                
                double d = current.first;
                int u = current.second;
                
                if (visited[u]) continue;
                
                #pragma omp critical
                {
                    visited[u] = true;
                }

                // Process adjacency list in parallel
                #pragma omp for reduction(||:globalChanges) nowait
                for (size_t i = 0; i < localAdj[u].size(); i++) {
                    int v = localAdj[u][i].first;
                    double w = localAdj[u][i].second;
                    double newDist = dist[u] + w;
                    
                    bool local_update = false;
                    #pragma omp critical
                    {
                        if (!visited[v] && newDist < dist[v]) {
                            dist[v] = newDist;
                            parent[v] = u;
                            local_update = true;
                        }
                    }
                    
                    if (local_update && (part[v] == rank || ghostVertices.count(v))) {
                        thread_updates.push_back({newDist, v});
                        globalChanges = true;
                    }
                }
            }
            
            // Add thread-local updates to priority queue
            if (!thread_updates.empty()) {
                #pragma omp critical
                {
                    for (const auto& update : thread_updates) {
                        pq.push(update);
                    }
                }
            }
        }

        // Prepare send buffers in parallel
        #pragma omp parallel
        {
            vector<vector<pair<int, double>>> thread_sendBuffers(size);
            for (int p = 0; p < size; ++p) {
                thread_sendBuffers[p].reserve(numVertices / (size * omp_get_num_threads()));
            }
            
            #pragma omp for nowait
            for (int v = 0; v < numVertices; ++v) {
                if (!visited[v] && dist[v] != INF && ghostVertices.count(v)) {
                    int targetRank = part[v];
                    thread_sendBuffers[targetRank].push_back({v, dist[v]});
                }
            }
            
            // Combine thread-local buffers
            #pragma omp critical
            {
                for (int p = 0; p < size; ++p) {
                    sendBuffers[p].insert(sendBuffers[p].end(), 
                                         thread_sendBuffers[p].begin(), 
                                         thread_sendBuffers[p].end());
                }
            }
        }

        // Optimize communication by using non-blocking sends
        vector<MPI_Request> requests;
        requests.reserve(2 * (size - 1));
        
        for (int p = 0; p < size; ++p) {
            if (p == rank) continue;
            int count = sendBuffers[p].size();
            MPI_Request req;
            MPI_Isend(&count, 1, MPI_INT, p, 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
            if (count > 0) {
                MPI_Isend(sendBuffers[p].data(), count * sizeof(pair<int, double>),
                          MPI_BYTE, p, 1, MPI_COMM_WORLD, &req);
                requests.push_back(req);
            }
        }

        // Process received updates in parallel
        for (int p = 0; p < size; ++p) {
            if (p == rank) continue;
            int count;
            MPI_Recv(&count, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (count > 0) {
                recvBuffers[p].resize(count);
                MPI_Recv(recvBuffers[p].data(), count * sizeof(pair<int, double>),
                         MPI_BYTE, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                #pragma omp parallel
                {
                    vector<pair<double, int>> thread_updates;
                    thread_updates.reserve(count / omp_get_num_threads() + 1);
                    
                    #pragma omp for reduction(||:globalChanges) nowait
                    for (int i = 0; i < count; i++) {
                        int v = recvBuffers[p][i].first;
                        double newDist = recvBuffers[p][i].second;
                        
                        bool local_update = false;
                        #pragma omp critical
                        {
                            if (!visited[v] && newDist < dist[v]) {
                                dist[v] = newDist;
                                parent[v] = -1;
                                local_update = true;
                            }
                        }
                        
                        if (local_update && part[v] == rank) {
                            thread_updates.push_back({newDist, v});
                            globalChanges = true;
                        }
                    }
                    
                    if (!thread_updates.empty()) {
                        #pragma omp critical
                        {
                            for (const auto& update : thread_updates) {
                                pq.push(update);
                            }
                        }
                    }
                }
            }
        }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

        // Clear send buffers for next iteration
        for (int p = 0; p < size; ++p) {
            sendBuffers[p].clear();
        }

        int localChanges = globalChanges ? 1 : 0;
        int globalFlag;
        MPI_Allreduce(&localChanges, &globalFlag, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        globalChanges = globalFlag > 0;
    }

    int rankVertices = 0;
    #pragma omp parallel for reduction(+:rankVertices)
    for (int v = 0; v < numVertices; ++v) {
        if (visited[v]) rankVertices++;
    }

    int totalVertices;
    MPI_Reduce(&rankVertices, &totalVertices, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        cout << "Reachable vertices: " << totalVertices << endl;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    string graphFile = argc > 1 ? argv[1] : "data.txt";
    string partitionFile = "partition.txt";

    double megaStart = MPI_Wtime();
    double start, end;
    double graphLoadingTime = 0.0;
    double initialSSSPTime = 0.0;
    double totalUpdateTime = 0.0;

    // Time graph loading
    if (rank == 0) {
        start = MPI_Wtime();
    }

    int numVertices = 0;
    long long numEdges = 0;
    unordered_map<int, int> idMap;
    vector<pair<int, int>> edgeList;

    if (rank == 0) {
        readGraph(graphFile, numVertices, numEdges, idMap, edgeList);
    }

    MPI_Bcast(&numVertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numEdges, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    int edgeListSize = edgeList.size();
    MPI_Bcast(&edgeListSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        edgeList.resize(edgeListSize);
    }
    vector<int> edgeU(edgeListSize), edgeV(edgeListSize);
    if (rank == 0) {
        #pragma omp parallel for
        for (size_t i = 0; i < edgeList.size(); ++i) {
            edgeU[i] = edgeList[i].first;
            edgeV[i] = edgeList[i].second;
        }
    }
    MPI_Bcast(edgeU.data(), edgeListSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(edgeV.data(), edgeListSize, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        #pragma omp parallel for
        for (int i = 0; i < edgeListSize; ++i) {
            edgeList[i] = {edgeU[i], edgeV[i]};
        }
    }

    int idMapSize = idMap.size();
    MPI_Bcast(&idMapSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    vector<int> idMapKeys, idMapValues;
    if (rank == 0) {
        idMapKeys.reserve(idMapSize);
        idMapValues.reserve(idMapSize);
        for (const auto& pair : idMap) {
            idMapKeys.push_back(pair.first);
            idMapValues.push_back(pair.second);
        }
    } else {
        idMapKeys.resize(idMapSize);
        idMapValues.resize(idMapSize);
    }
    MPI_Bcast(idMapKeys.data(), idMapSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(idMapValues.data(), idMapSize, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        idMap.clear();
        #pragma omp parallel
        {
            unordered_map<int, int> thread_idMap;
            #pragma omp for nowait
            for (int i = 0; i < idMapSize; ++i) {
                thread_idMap[idMapKeys[i]] = idMapValues[i];
            }
            
            #pragma omp critical
            {
                idMap.insert(thread_idMap.begin(), thread_idMap.end());
            }
        }
    }

    if (rank == 0) {
        end = MPI_Wtime();
        graphLoadingTime = (end - start) * 1000.0;
    }

    vector<idx_t> xadj, adjncy, part;
    convertToCSR(edgeList, numVertices, xadj, adjncy);
    int nparts = size;
    partitionGraph(numVertices, xadj, adjncy, nparts, part);

    if (rank == 0) {
        ofstream out(partitionFile);
        for (size_t i = 0; i < part.size(); ++i) {
            out << i << " " << part[i] << endl;
        }
        out.close();
        cout << "Partitioning saved to " << partitionFile << endl;

        vector<int> partSizes(nparts, 0);
        #pragma omp parallel for
        for (size_t i = 0; i < part.size(); i++) {
            #pragma omp atomic
            partSizes[part[i]]++;
        }
        cout << "Partition sizes:" << endl;
        for (int i = 0; i < nparts; ++i) {
            cout << "Partition " << i << ": " << partSizes[i] << " vertices" << endl;
        }
    }

    vector<vector<pair<int, double>>> localAdj;
    vector<int> localVertices;
    unordered_set<int> ghostVertices;
    distributeSubgraphs(edgeList, part, rank, size, numVertices,
                       localAdj, localVertices, ghostVertices);

    vector<pair<Edge, bool>> changes;
    if (rank == 0) {
        initializeUpdates(changes, numVertices, edgeList);
    }
    int changesSize = changes.size();
    MPI_Bcast(&changesSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        changes.resize(changesSize);
    }
    vector<int> changeU(changesSize), changeV(changesSize);
    vector<double> changeW(changesSize);
    vector<char> changeOp(changesSize);
    if (rank == 0) {
        #pragma omp parallel for
        for (size_t i = 0; i < changes.size(); ++i) {
            changeU[i] = changes[i].first.u;
            changeV[i] = changes[i].first.v;
            changeW[i] = changes[i].first.w;
            changeOp[i] = changes[i].second ? 'I' : 'D';
        }
    }
    MPI_Bcast(changeU.data(), changesSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(changeV.data(), changesSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(changeW.data(), changesSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(changeOp.data(), changesSize, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        #pragma omp parallel for
        for (int i = 0; i < changesSize; ++i) {
            changes[i] = {Edge{changeU[i], changeV[i], changeW[i]}, changeOp[i] == 'I'};
        }
    }

    vector<double> dist;
    vector<int> parent;

    // Time initial SSSP
    if (rank == 0) {
        start = MPI_Wtime();
    }

    computeInitialSSSP(localAdj, localVertices, ghostVertices, part, rank, size, numVertices, dist, parent, 0);

    if (rank == 0) {
        end = MPI_Wtime();
        initialSSSPTime = (end - start) * 1000.0;
    }

    // Process updates with timing
    int changeIndex = 1;
    totalUpdateTime = 0.0;
    for (const auto& change : changes) {
        if (rank == 0) {
            start = MPI_Wtime();
        }

        updateSSSP(change.first, change.second, localAdj, localVertices, ghostVertices, part, rank, size, numVertices, dist, parent);

        if (rank == 0) {
            end = MPI_Wtime();
            totalUpdateTime += (end - start) * 1000.0;
        }
        changeIndex++;
    }

    // Final execution time summary
    if (rank == 0) {
        double megaEnd = MPI_Wtime();
        double totalExecutionTime = (megaEnd - megaStart) * 1000.0;
        
        cout << "\nPerformance Summary:" << endl;
        cout << "------------------" << endl;
        cout << "Graph Loading:     " << graphLoadingTime << " ms" << endl;
        cout << "Initial SSSP:      " << initialSSSPTime << " ms" << endl;
        cout << "Total Updates:     " << totalUpdateTime << " ms" << endl;
        cout << "Avg Update Time:   " << totalUpdateTime / changes.size() << " ms" << endl;
        cout << "Total Runtime:     " << totalExecutionTime << " ms" << endl;
        cout << "------------------" << endl;
    }

    MPI_Finalize();
    return 0;
}