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
