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

using namespace std;

// Edge structure
struct Edge {
    int u, v;
    double w;
};

// Read graph
//READ GRAPH
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
    for (auto v : vertices) {
        idMap[v] = newId++;
    }

    for (auto& e : edgeList) {
        e.first = idMap[e.first];
        e.second = idMap[e.second];
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
    for (const auto& edge : edgeList) {
        adj[edge.first].push_back(edge.second);
        adj[edge.second].push_back(edge.first); // METIS requires undirected
    }

    xadj.resize(numVertices + 1);
    adjncy.clear();
    idx_t offset = 0;
    xadj[0] = 0;
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

    for (int v = 0; v < numVertices; ++v) {
        if (part[v] == rank) {
            localVertices.push_back(v);
        }
    }

    for (const auto& edge : edgeList) {
        int u = edge.first, v = edge.second;
        double w = 1.0; // Default weight
        if (part[u] == rank) {
            localAdj[u].push_back({v, w});
            if (part[v] != rank) ghostVertices.insert(v);
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
    for (const auto& edge : edgeList) {
        long long edgeKey = static_cast<long long>(edge.first) * numVertices + edge.second;
        existingEdges.insert(edgeKey);
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

// Parallel incremental SSSP update
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
                localAdj[e.u].push_back({e.v, e.w});
                if (part[e.v] != rank) ghostVertices.insert(e.v);
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
                auto& adj = localAdj[e.u];
                adj.erase(remove_if(adj.begin(), adj.end(),
                                   [v = e.v](const auto& p) { return p.first == v; }),
                         adj.end());
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

            for (const auto& [n, w] : localAdj[z]) {
                double newDist = (dist[z] == numeric_limits<double>::infinity())
                               ? numeric_limits<double>::infinity()
                               : dist[z] + w;
                if (dist[n] > newDist) {
                    dist[n] = newDist;
                    parent[n] = z;
                    if (part[n] == rank || ghostVertices.count(n)) {
                        affected.insert(n);
                        pq.push({dist[n], n});
                        globalChanges = true;
                    }
                }
            }
        }

        // Collect updates for ghost vertices
        for (int p = 0; p < size; ++p) {
            sendBuffers[p].clear();
        }
        for (int v = 0; v < numVertices; ++v) {
            if (affected.count(v) && ghostVertices.count(v)) {
                int targetRank = part[v];
                sendBuffers[targetRank].push_back({v, dist[v]});
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
                for (const auto& [v, newDist] : recvBuffers[p]) {
                    if (newDist < dist[v]) {
                        dist[v] = newDist;
                        parent[v] = -1; // Parent may be in another partition
                        if (part[v] == rank) {
                            affected.insert(v);
                            pq.push({dist[v], v});
                            globalChanges = true;
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

            for (const auto& [v, w] : localAdj[u]) {
                double newDist = dist[u] + w;
                if (!visited.count(v) && newDist < dist[v]) {
                    dist[v] = newDist;
                    parent[v] = u;
                    if (part[v] == rank || ghostVertices.count(v)) {
                        pq.push({newDist, v});
                        globalChanges = true;
                    }
                }
            }
        }

        for (int p = 0; p < size; ++p) {
            sendBuffers[p].clear();
        }
        for (int v = 0; v < numVertices; ++v) {
            if (!visited.count(v) && dist[v] != INF && ghostVertices.count(v)) {
                int targetRank = part[v];
                sendBuffers[targetRank].push_back({v, dist[v]});
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
                for (const auto& [v, newDist] : recvBuffers[p]) {
                    if (!visited.count(v) && newDist < dist[v]) {
                        dist[v] = newDist;
                        parent[v] = -1; // Parent may be in another partition
                        if (part[v] == rank) {
                            pq.push({newDist, v});
                            globalChanges = true;
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

    int rankVertices = visited.size();
    int totalVertices;
    MPI_Reduce(&rankVertices, &totalVertices, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        cout << "Reachable vertices: " << totalVertices << endl;
    }
}
//MAIN FUNCTION 
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    string graphFile = argc > 1 ? argv[1] : "data.txt";
    string partitionFile = "partition.txt";

    double megaStart = MPI_Wtime(); // Total execution time start
    double start, end;

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
        for (size_t i = 0; i < edgeList.size(); ++i) {
            edgeU[i] = edgeList[i].first;
            edgeV[i] = edgeList[i].second;
        }
    }
    MPI_Bcast(edgeU.data(), edgeListSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(edgeV.data(), edgeListSize, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
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
        for (int i = 0; i < idMapSize; ++i) {
            idMap[idMapKeys[i]] = idMapValues[i];
        }
    }

    if (rank == 0) {
        end = MPI_Wtime();
        double graphLoadingTime = (end - start) * 1000.0; // Convert to ms
        cout << "Graph loading time: " << graphLoadingTime << " ms" << endl;
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
        for (idx_t p : part) {
            partSizes[p]++;
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

    // Initial SSSP from source vertex 0
    computeInitialSSSP(localAdj, localVertices, ghostVertices, part, rank, size, numVertices, dist, parent, 0);

    if (rank == 0) {
        end = MPI_Wtime();
        double initialSSSPTime = (end - start) * 1000.0; // Convert to ms
        cout << "Initial SSSP computation time: " << initialSSSPTime << " ms" << endl;
    }

    // Process updates with timing
    int changeIndex = 1;
    double totalUpdateTime = 0.0;
    for (const auto& change : changes) {
        if (rank == 0) {
            start = MPI_Wtime();
        }

        updateSSSP(change.first, change.second, localAdj, localVertices, ghostVertices, part, rank, size, numVertices, dist, parent);

        if (rank == 0) {
            end = MPI_Wtime();
            double updateDuration = (end - start) * 1000.0; // Convert to ms
            totalUpdateTime += updateDuration;

            if (changeIndex % 10000 == 0 || changeIndex == 1 || changeIndex == changes.size()) {
                cout << "\n---------------------------------" << endl;
                cout << "Change " << changeIndex << ":" << endl;
                cout << "Update time: " << updateDuration << " ms" << endl;
            }
        }
        changeIndex++;
    }

    if (rank == 0) {
        cout << "\nDynamic Updates Summary:" << endl;
        cout << "Total update time: " << totalUpdateTime << " ms" << endl;
        cout << "Average update time: " << totalUpdateTime / changes.size() << " ms" << endl;
    }

    // Gather distances to rank 0
    vector<double> globalDist;
    if (rank == 0) {
        globalDist.resize(numVertices, numeric_limits<double>::infinity());
    }

    // Prepare local distances
    vector<int> sendVertices;
    vector<double> sendDistances;
    for (int v : localVertices) {
        if (dist[v] != numeric_limits<double>::infinity()) {
            sendVertices.push_back(v);
            sendDistances.push_back(dist[v]);
        }
    }
    int localDistCount = sendVertices.size();

    // Gather distance counts
    vector<int> distCounts(size);
    MPI_Gather(&localDistCount, 1, MPI_INT, distCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Prepare for MPI_Gatherv
    vector<int> recvCounts(size), displs(size);
    int totalRecv = 0;
    if (rank == 0) {
        for (int p = 0; p < size; ++p) {
            recvCounts[p] = distCounts[p];
            displs[p] = totalRecv;
            totalRecv += distCounts[p];
        }
    }

    // Gather vertices
    vector<int> allVertices(totalRecv);
    MPI_Gatherv(sendVertices.data(), localDistCount, MPI_INT,
                allVertices.data(), recvCounts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    // Gather distances
    vector<double> allDistances(totalRecv);
    MPI_Gatherv(sendDistances.data(), localDistCount, MPI_DOUBLE,
                allDistances.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Process received distances
    if (rank == 0) {
        for (int i = 0; i < totalRecv; ++i) {
            int v = allVertices[i];
            double d = allDistances[i];
            globalDist[v] = min(globalDist[v], d);
        }

        // Write distances
        ofstream out("distances.txt");
        for (int v = 0; v < numVertices; ++v) {
            if (globalDist[v] != numeric_limits<double>::infinity()) {
                out << v << " " << globalDist[v] << endl;
            }
        }
        out.close();
        cout << "Distances saved to distances.txt" << endl;
    }

    // Final execution time summary
    if (rank == 0) {
        double megaEnd = MPI_Wtime();
        double totalExecutionTime = (megaEnd - megaStart) * 1000000.0; // Convert to microseconds
        cout << "\nFinal Execution Time Summary:" << endl;
        cout << "-----------------------------" << endl;
        cout << "Graph loading time:          " << (end - start) * 1000.0 << " ms" << endl;
        cout << "Initial SSSP computation:    " << (end - start) * 1000.0 << " ms" << endl;
        cout << "Total dynamic updates:       " << totalUpdateTime << " ms" << endl;
       // cout << "Average dynamic update:      " << totalUpdateTime / changes.size() << " ms" << endl;
        //cout << "Total time of execution is   " << totalExecutionTime << " s" << endl;
    }

    MPI_Finalize();
    return 0;
}
