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
#include <omp.h>  // Added OpenMP header

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

    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            if (line.find("Nodes:") != string::npos) {
                sscanf(line.c_str(), "# Nodes: %d Edges: %lld", &expectedNodes, &expectedEdges);
            }
            continue;
        }
        stringstream ss(line);
        int u, v;
        if (ss >> u >> v) {
            vertices.insert(u);
            vertices.insert(v);
            edgeList.push_back({u, v});
            edgesRead++;
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (edgesRead % 1000000 == 0 && rank == 0) {
                cout << "Scanned " << edgesRead << " edges..." << endl;
            }
        } else {
            cerr << "Warning: Skipping malformed line: " << line << endl;
        }
    }
    file.close();

    if (vertices.empty()) {
        cerr << "Error: No valid vertices found in " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    numVertices = vertices.size();
    int newId = 0;
    // Use OpenMP to parallelize ID mapping creation
    vector<pair<int, int>> vertex_pairs;
    vertex_pairs.reserve(vertices.size());
    for (auto v : vertices) {
        vertex_pairs.push_back({v, newId++});
    }
    
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < vertex_pairs.size(); i++) {
            #pragma omp critical
            {
                idMap[vertex_pairs[i].first] = vertex_pairs[i].second;
            }
        }
    }

    // Parallelize edge list remapping
    #pragma omp parallel for schedule(dynamic, 1000)
    for (size_t i = 0; i < edgeList.size(); i++) {
        edgeList[i].first = idMap[edgeList[i].first];
        edgeList[i].second = idMap[edgeList[i].second];
    }

    numEdges = edgesRead;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        cout << "Detected " << numVertices << " nodes (" << vertices.size() << " unique vertices)" << endl;
        cout << "Read " << edgesRead << " edges" << endl;
    }
}

// Convert to CSR for METIS
void convertToCSR(const vector<pair<int, int>>& edgeList, int numVertices,
                  vector<idx_t>& xadj, vector<idx_t>& adjncy) {
    vector<vector<int>> adj(numVertices);
    
    // Parallelize adjacency list creation
    #pragma omp parallel
    {
        // Create thread-local adjacency lists to avoid locks
        vector<vector<int>> local_adj(numVertices);
        
        #pragma omp for schedule(dynamic, 1000)
        for (size_t i = 0; i < edgeList.size(); i++) {
            const auto& edge = edgeList[i];
            local_adj[edge.first].push_back(edge.second);
            local_adj[edge.second].push_back(edge.first); // METIS requires undirected
        }
        
        // Merge thread-local results
        #pragma omp critical
        {
            for (int u = 0; u < numVertices; ++u) {
                adj[u].insert(adj[u].end(), local_adj[u].begin(), local_adj[u].end());
            }
        }
    }

    xadj.resize(numVertices + 1);
    adjncy.clear();
    xadj[0] = 0;
    
    // Pre-calculate the size of adjncy
    idx_t total_edges = 0;
    for (int u = 0; u < numVertices; ++u) {
        total_edges += adj[u].size();
    }
    adjncy.reserve(total_edges);
    
    // Fill xadj and adjncy
    idx_t offset = 0;
    for (int u = 0; u < numVertices; ++u) {
        for (int v : adj[u]) {
            adjncy.push_back(v);
        }
        offset += adj[u].size();
        xadj[u + 1] = offset;
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

    // Parallelize local vertices identification
    vector<bool> isLocalVertex(numVertices, false);
    #pragma omp parallel for schedule(static)
    for (int v = 0; v < numVertices; ++v) {
        if (part[v] == rank) {
            isLocalVertex[v] = true;
        }
    }
    
    // Collect local vertices
    for (int v = 0; v < numVertices; ++v) {
        if (isLocalVertex[v]) {
            localVertices.push_back(v);
        }
    }

    // Use thread-local containers to avoid critical sections
    vector<unordered_set<int>> thread_ghost_vertices;
    vector<vector<vector<pair<int, double>>>> thread_local_adj;
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        
        // Initialize thread-local containers
        #pragma omp single
        {
            thread_ghost_vertices.resize(num_threads);
            thread_local_adj.resize(num_threads, vector<vector<pair<int, double>>>(numVertices));
        }
        
        // Divide edge list among threads
        #pragma omp for schedule(dynamic, 1000)
        for (size_t i = 0; i < edgeList.size(); ++i) {
            int u = edgeList[i].first;
            int v = edgeList[i].second;
            double w = 1.0; // Default weight
            
            if (part[u] == rank) {
                thread_local_adj[thread_id][u].push_back({v, w});
                if (part[v] != rank) 
                    thread_ghost_vertices[thread_id].insert(v);
            }
        }
    }
    
    // Merge thread-local results
    for (size_t t = 0; t < thread_ghost_vertices.size(); ++t) {
        ghostVertices.insert(thread_ghost_vertices[t].begin(), thread_ghost_vertices[t].end());
        
        for (int u = 0; u < numVertices; ++u) {
            if (!thread_local_adj[t][u].empty()) {
                localAdj[u].insert(localAdj[u].end(), 
                                 thread_local_adj[t][u].begin(), 
                                 thread_local_adj[t][u].end());
            }
        }
    }
}

// Initialize updates (100,000 random operations)
void initializeUpdates(vector<pair<Edge, bool>>& changes, int numVertices,
                       const vector<pair<int, int>>& edgeList) {
    int NUM_CHANGES = 100;
    changes.clear();
    changes.reserve(NUM_CHANGES);
    unordered_set<long long> existingEdges;
    
    // Parallelize building existing edges set
    #pragma omp parallel
    {
        unordered_set<long long> local_existing_edges;
        
        #pragma omp for schedule(dynamic, 1000)
        for (size_t i = 0; i < edgeList.size(); ++i) {
            const auto& edge = edgeList[i];
            long long edgeKey = static_cast<long long>(edge.first) * numVertices + edge.second;
            local_existing_edges.insert(edgeKey);
        }
        
        #pragma omp critical
        {
            existingEdges.insert(local_existing_edges.begin(), local_existing_edges.end());
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

// Parallel incremental SSSP update with OpenMP
void updateSSSP(const Edge& e, bool isInsertion,
                vector<vector<pair<int, double>>>& localAdj,
                const vector<int>& localVertices,
                unordered_set<int>& ghostVertices,
                const vector<idx_t>& part, int rank, int size, int numVertices,
                vector<double>& dist, vector<int>& parent) {
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
    unordered_set<int> affected;

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
                affected.insert(x);
                pq.push({dist[x], x});
            }
        } else {
            if (part[e.u] == rank && isEdgeInSSSPTree(dist, parent, e.u, e.v)) {
                if (parent[x] == y) {
                    dist[x] = numeric_limits<double>::infinity();
                    parent[x] = -1;
                    affected.insert(x);
                    pq.push({dist[x], x});
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

    vector<vector<pair<int, double>>> sendBuffers(size);
    vector<vector<pair<int, double>>> recvBuffers(size);

    bool globalChanges = true;
    while (globalChanges) {
        globalChanges = false;

        while (!pq.empty()) {
            double d = pq.top().first;
            int z = pq.top().second;
            pq.pop();
            if (affected.find(z) == affected.end()) continue;
            affected.erase(z);

            // Create local copies to avoid data race
            vector<pair<int, double>> neighbors;
            for (const auto& neighbor : localAdj[z]) {
                neighbors.push_back(neighbor);
            }

            // Process neighbors in parallel
            #pragma omp parallel
            {
                unordered_set<int> thread_affected;
                vector<P> thread_updates;
                
                #pragma omp for nowait
                for (size_t i = 0; i < neighbors.size(); i++) {
                    int n = neighbors[i].first;
                    double w = neighbors[i].second;
                    
                    double newDist = (dist[z] == numeric_limits<double>::infinity())
                                   ? numeric_limits<double>::infinity()
                                   : dist[z] + w;
                                   
                    if (newDist < dist[n]) {
                        #pragma omp critical
                        {
                            // Check again to avoid race conditions
                            if (newDist < dist[n]) {
                                dist[n] = newDist;
                                parent[n] = z;
                                
                                if (part[n] == rank || ghostVertices.count(n)) {
                                    thread_affected.insert(n);
                                    thread_updates.push_back({dist[n], n});
                                    globalChanges = true;
                                }
                            }
                        }
                    }
                }
                
                // Merge thread results
                #pragma omp critical
                {
                    affected.insert(thread_affected.begin(), thread_affected.end());
                    for (const auto& update : thread_updates) {
                        pq.push(update);
                    }
                }
            }
        }

        // Collect updates for ghost vertices in parallel
        for (int p = 0; p < size; ++p) {
            sendBuffers[p].clear();
        }
        
        vector<vector<pair<int, double>>> thread_send_buffers(omp_get_max_threads(), vector<pair<int, double>>(size));
        
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            vector<vector<pair<int, double>>> local_send_buffers(size);
            
            #pragma omp for schedule(dynamic, 64)
            for (int v = 0; v < numVertices; ++v) {
                if (affected.count(v) && ghostVertices.count(v)) {
                    int targetRank = part[v];
                    local_send_buffers[targetRank].push_back({v, dist[v]});
                }
            }
            
            // Merge thread results
            #pragma omp critical
            {
                for (int p = 0; p < size; ++p) {
                    sendBuffers[p].insert(sendBuffers[p].end(), 
                                        local_send_buffers[p].begin(), 
                                        local_send_buffers[p].end());
                }
            }
        }

        // Communicate updates
        vector<MPI_Request> requests;
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

        for (int p = 0; p < size; ++p) {
            if (p == rank) continue;
            int count;
            MPI_Recv(&count, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (count > 0) {
                recvBuffers[p].resize(count);
                MPI_Recv(recvBuffers[p].data(), count * sizeof(pair<int, double>),
                         MPI_BYTE, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // Process received data in parallel
                #pragma omp parallel
                {
                    unordered_set<int> thread_affected;
                    vector<P> thread_updates;
                    
                    #pragma omp for schedule(dynamic, 64)
                    for (int i = 0; i < count; i++) {
                        int v = recvBuffers[p][i].first;
                        double newDist = recvBuffers[p][i].second;
                        
                        if (newDist < dist[v]) {
                            #pragma omp critical
                            {
                                if (newDist < dist[v]) {
                                    dist[v] = newDist;
                                    parent[v] = -1; // Parent may be in another partition
                                    
                                    if (part[v] == rank) {
                                        thread_affected.insert(v);
                                        thread_updates.push_back({dist[v], v});
                                        globalChanges = true;
                                    }
                                }
                            }
                        }
                    }
                    
                    // Merge thread results
                    #pragma omp critical
                    {
                        affected.insert(thread_affected.begin(), thread_affected.end());
                        for (const auto& update : thread_updates) {
                            pq.push(update);
                        }
                    }
                }
            }
        }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

        int localChanges = globalChanges ? 1 : 0;
        int globalFlag;
        MPI_Allreduce(&localChanges, &globalFlag, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        globalChanges = globalFlag > 0;
    }
}

// Compute initial SSSP with OpenMP parallelism
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
    unordered_set<int> visited;
    if (part[source] == rank) {
        pq.push({0, source});
    }

    vector<vector<pair<int, double>>> sendBuffers(size);
    vector<vector<pair<int, double>>> recvBuffers(size);

    bool globalChanges = true;
    while (globalChanges) {
        globalChanges = false;

        while (!pq.empty()) {
            double d = pq.top().first;
            int u = pq.top().second;
            pq.pop();
            if (visited.count(u)) continue;
            visited.insert(u);

            // Create a local copy of neighbors to avoid data race
            vector<pair<int, double>> neighbors;
            for (const auto& neighbor : localAdj[u]) {
                neighbors.push_back(neighbor);
            }

            // Process neighbors in parallel
            #pragma omp parallel
            {
                vector<P> thread_updates;
                unordered_set<int> thread_visited;
                
                #pragma omp for nowait
                for (size_t i = 0; i < neighbors.size(); i++) {
                    int v = neighbors[i].first;
                    double w = neighbors[i].second;
                    
                    double newDist = dist[u] + w;
                    bool should_update = false;
                    
                    #pragma omp critical
                    {
                        if (!visited.count(v) && newDist < dist[v]) {
                            dist[v] = newDist;
                            parent[v] = u;
                            should_update = true;
                        }
                    }
                    
                    if (should_update && (part[v] == rank || ghostVertices.count(v))) {
                        thread_updates.push_back({newDist, v});
                        globalChanges = true;
                    }
                }
                
                // Merge thread results
                #pragma omp critical
                {
                    for (const auto& update : thread_updates) {
                        pq.push(update);
                    }
                    visited.insert(thread_visited.begin(), thread_visited.end());
                }
            }
        }

        // Prepare send buffers in parallel
        for (int p = 0; p < size; ++p) {
            sendBuffers[p].clear();
        }
        
        vector<vector<vector<pair<int, double>>>> thread_send_buffers(omp_get_max_threads(), 
                                                                    vector<vector<pair<int, double>>>(size));
        
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            
            #pragma omp for schedule(dynamic, 64)
            for (int v = 0; v < numVertices; ++v) {
                if (!visited.count(v) && dist[v] != INF && ghostVertices.count(v)) {
                    int targetRank = part[v];
                    thread_send_buffers[thread_id][targetRank].push_back({v, dist[v]});
                }
            }
        }
        
        // Merge thread results
        for (int t = 0; t < omp_get_max_threads(); ++t) {
            for (int p = 0; p < size; ++p) {
                sendBuffers[p].insert(sendBuffers[p].end(), 
                                    thread_send_buffers[t][p].begin(), 
                                    thread_send_buffers[t][p].end());
            }
        }

        vector<MPI_Request> requests;
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

        for (int p = 0; p < size; ++p) {
            if (p == rank) continue;
            int count;
            MPI_Recv(&count, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (count > 0) {
                recvBuffers[p].resize(count);
                MPI_Recv(recvBuffers[p].data(), count * sizeof(pair<int, double>),
                         MPI_BYTE, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // Process received updates in parallel
                #pragma omp parallel
                {
                    vector<P> thread_updates;
                    
                    #pragma omp for schedule(dynamic, 64)
                    for (int i = 0; i < count; i++) {
                        int v = recvBuffers[p][i].first;
                        double newDist = recvBuffers[p][i].second;
                        
                        bool should_update = false;
                        #pragma omp critical
                        {
                            if (newDist < dist[v]) {
                                dist[v] = newDist;
                                parent[v] = -1; // Parent in another partition
                                should_update = true;
                            }
                        }
                        
                        if (should_update && part[v] == rank) {
                            thread_updates.push_back({newDist, v});
                            globalChanges = true;
                        }
                    }
                    
                    // Merge thread results
                    #pragma omp critical
                    {
                        for (const auto& update : thread_updates) {
                            pq.push(update);
                        }
                    }
                }
            }
        }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

        int localChanges = globalChanges ? 1 : 0;
        int globalFlag;
        MPI_Allreduce(&localChanges, &globalFlag, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        globalChanges = globalFlag > 0;
    }
}

// Print SSSP results (for debugging)
void printSSSP(const vector<double>& dist, const vector<int>& parent, int source, int rank, int numVertices) {
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    
    for (int r = 0; r < worldSize; ++r) {
        if (r == rank) {
            cout << "Process " << rank << " SSSP results from source " << source << ":" << endl;
            for (int v = 0; v < numVertices; ++v) {
                if (dist[v] < numeric_limits<double>::infinity()) {
                    cout << "  Vertex " << v << ": dist = " << dist[v] << ", parent = " << parent[v] << endl;
                }
            }
            cout << flush;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

// Verify SSSP correctness by comparing with sequential Dijkstra
bool verifySSSP(const vector<pair<int, int>>& edgeList, const vector<double>& parallelDist, 
                int numVertices, int source, int rank) {
    if (rank != 0) return true;
    
    cout << "Verifying SSSP results..." << endl;
    
    // Build adjacency list for sequential algorithm
    vector<vector<pair<int, double>>> adj(numVertices);
    for (const auto& edge : edgeList) {
        int u = edge.first;
        int v = edge.second;
        double w = 1.0; // Default weight
        adj[u].push_back({v, w});
        adj[v].push_back({u, w}); // Undirected graph
    }
    
    // Run sequential Dijkstra
    vector<double> seqDist(numVertices, numeric_limits<double>::infinity());
    vector<int> seqParent(numVertices, -1);
    vector<bool> visited(numVertices, false);
    
    seqDist[source] = 0;
    
    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;
    pq.push({0, source});
    
    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();
        
        if (visited[u]) continue;
        visited[u] = true;
        
        for (const auto& edge : adj[u]) {
            int v = edge.first;
            double w = edge.second;
            
            if (!visited[v] && seqDist[u] + w < seqDist[v]) {
                seqDist[v] = seqDist[u] + w;
                seqParent[v] = u;
                pq.push({seqDist[v], v});
            }
        }
    }
    
    // Compare results
    bool correct = true;
    for (int v = 0; v < numVertices; ++v) {
        if (abs(seqDist[v] - parallelDist[v]) > 1e-6 && 
            !(seqDist[v] == numeric_limits<double>::infinity() && 
              parallelDist[v] == numeric_limits<double>::infinity())) {
            cout << "Mismatch at vertex " << v << ": sequential = " << seqDist[v] 
                 << ", parallel = " << parallelDist[v] << endl;
            correct = false;
        }
    }
    
    if (correct) {
        cout << "SSSP verification passed!" << endl;
    } else {
        cout << "SSSP verification failed!" << endl;
    }
    
    return correct;
}

// Benchmark function
void benchmarkSSP(vector<pair<Edge, bool>>& changes, 
                 vector<vector<pair<int, double>>>& localAdj,
                 const vector<int>& localVertices,
                 unordered_set<int>& ghostVertices,
                 const vector<idx_t>& part, int rank, int size, int numVertices,
                 vector<double>& dist, vector<int>& parent, int source) {
    // Time sequential updates
    double seqTime = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double startSeq = MPI_Wtime();
    
    for (const auto& change : changes) {
        const Edge& e = change.first;
        bool isInsertion = change.second;
        
        // Turn off OpenMP for sequential timing
        omp_set_num_threads(1);
        updateSSSP(e, isInsertion, localAdj, localVertices, ghostVertices, 
                  part, rank, size, numVertices, dist, parent);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double endSeq = MPI_Wtime();
    seqTime = endSeq - startSeq;
    
    // Reset SSSP
    computeInitialSSSP(localAdj, localVertices, ghostVertices, part, rank, size, numVertices, dist, parent, source);
    
    // Time parallel updates
    double parTime = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double startPar = MPI_Wtime();
    
    // Set OpenMP threads to maximum
    omp_set_num_threads(omp_get_max_threads());
    
    for (const auto& change : changes) {
        const Edge& e = change.first;
        bool isInsertion = change.second;
        
        updateSSSP(e, isInsertion, localAdj, localVertices, ghostVertices, 
                  part, rank, size, numVertices, dist, parent);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double endPar = MPI_Wtime();
    parTime = endPar - startPar;
    
    if (rank == 0) {
        cout << "Sequential update time: " << seqTime << " seconds" << endl;
        cout << "Parallel update time: " << parTime << " seconds" << endl;
        cout << "Speedup: " << seqTime / parTime << "x" << endl;
    }
}

// Main function
int main(int argc, char* argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        cerr << "Warning: The MPI implementation does not support MPI_THREAD_MULTIPLE" << endl;
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    string filename = "data.txt";
    int source = 0; // Default source vertex

    if (argc >= 3) {
        source = atoi(argv[2]);
    }

    int numVertices;
    long long numEdges;
    unordered_map<int, int> idMap;
    vector<pair<int, int>> edgeList;

    // Timing variables
    double startTime, endTime;
    
    if (rank == 0) {
        cout << "Reading graph from " << filename << endl;
    }
    startTime = MPI_Wtime();
    readGraph(filename, numVertices, numEdges, idMap, edgeList);
    endTime = MPI_Wtime();
    if (rank == 0) {
        cout << "Graph reading completed in " << (endTime - startTime) << " seconds" << endl;
    }

    vector<idx_t> xadj, adjncy;
    if (rank == 0) {
        cout << "Converting to CSR format for METIS..." << endl;
    }
    startTime = MPI_Wtime();
    convertToCSR(edgeList, numVertices, xadj, adjncy);
    endTime = MPI_Wtime();
    if (rank == 0) {
        cout << "CSR conversion completed in " << (endTime - startTime) << " seconds" << endl;
    }

    vector<idx_t> part(numVertices);
    if (rank == 0) {
        cout << "Partitioning graph with METIS into " << size << " parts..." << endl;
    }
    startTime = MPI_Wtime();
    partitionGraph(numVertices, xadj, adjncy, size, part);
    endTime = MPI_Wtime();
    if (rank == 0) {
        cout << "Graph partitioning completed in " << (endTime - startTime) << " seconds" << endl;
    }

    vector<vector<pair<int, double>>> localAdj;
    vector<int> localVertices;
    unordered_set<int> ghostVertices;

    if (rank == 0) {
        cout << "Distributing subgraphs to processes..." << endl;
    }
    startTime = MPI_Wtime();
    distributeSubgraphs(edgeList, part, rank, size, numVertices, localAdj, localVertices, ghostVertices);
    endTime = MPI_Wtime();
    if (rank == 0) {
        cout << "Subgraph distribution completed in " << (endTime - startTime) << " seconds" << endl;
        cout << "Starting with " << localVertices.size() << " local vertices and " 
             << ghostVertices.size() << " ghost vertices" << endl;
    }

    vector<double> dist;
    vector<int> parent;

    if (rank == 0) {
        cout << "Computing initial SSSP from source " << source << "..." << endl;
    }
    startTime = MPI_Wtime();
    computeInitialSSSP(localAdj, localVertices, ghostVertices, part, rank, size, numVertices, dist, parent, source);
    endTime = MPI_Wtime();
    if (rank == 0) {
        cout << "Initial SSSP computation completed in " << (endTime - startTime) << " seconds" << endl;
    }

    // Verify SSSP results (optional)
    if (verifySSSP(edgeList, dist, numVertices, source, rank)) {
        if (rank == 0) cout << "Initial SSSP computation verified successfully" << endl;
    }

    // Preparing updates
    vector<pair<Edge, bool>> changes;
    if (rank == 0) {
        cout << "Generating random graph updates..." << endl;
    }
    startTime = MPI_Wtime();
    initializeUpdates(changes, numVertices, edgeList);
    endTime = MPI_Wtime();
    if (rank == 0) {
        cout << "Updates generation completed in " << (endTime - startTime) << " seconds" << endl;
    }

    // Benchmark updates
    if (rank == 0) {
        cout << "Benchmarking incremental SSSP updates..." << endl;
    }
    benchmarkSSP(changes, localAdj, localVertices, ghostVertices, 
                part, rank, size, numVertices, dist, parent, source);

    MPI_Finalize();
    return 0;
}