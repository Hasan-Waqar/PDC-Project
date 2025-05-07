#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;

// Edge structure for dynamic changes
struct Edge {
    int u, v;
    double weight;
    Edge(int _u, int _v, double _w) : u(_u), v(_v), weight(_w) {}
};

// Graph class using adjacency list
class Graph {
public:
    int numVertices;
    long long numEdges;
    vector<vector<pair<int, double> > > adj;

    Graph(int n) : numVertices(n), numEdges(0), adj(n) {}

    void addEdge(int u, int v, double w) {
        if (u >= numVertices || v >= numVertices) {
            int newSize = u > v ? u + 1 : v + 1;
            adj.resize(newSize);
            numVertices = newSize;
        }
        adj[u].push_back(pair<int, double>(v, w));
        numEdges++;
    }

    bool removeEdge(int u, int v) {
        if (u >= numVertices || v >= numVertices) {
            return false;
        }
        for (vector<pair<int, double> >::iterator it = adj[u].begin(); it != adj[u].end(); ++it) {
            if (it->first == v) {
                adj[u].erase(it);
                numEdges--;
                return true;
            }
        }
        return false;
    }

    void printGraphSummary() const {
        cout << "Graph Summary:" << endl;
        cout << "Nodes: " << numVertices << ", Edges: " << numEdges << endl;
        cout << "Sample adjacency list (first 5 vertices):" << endl;
        int limit = numVertices < 5 ? numVertices : 5;
        for (int u = 0; u < limit; ++u) {
            cout << "Vertex " << u << ": ";
            for (vector<pair<int, double> >::const_iterator neighbor = adj[u].begin(); neighbor != adj[u].end(); ++neighbor) {
                cout << "(" << neighbor->first << ", " << neighbor->second << ") ";
            }
            cout << endl;
        }
    }
};

// SSSP Tree storing distances and parents
class SSSPTree {
public:
    vector<double> dist;
    vector<int> parent;
    int source;

    SSSPTree(int n, int s) : dist(n, numeric_limits<double>::infinity()), parent(n, -1), source(s) {
        dist[s] = 0.0;
    }

    void resize(int n) {
        dist.resize(n, numeric_limits<double>::infinity());
        parent.resize(n, -1);
        if (source < n) {
            dist[source] = 0.0;
        }
    }

    void printTree(int maxVertices = 10) const {
        cout << "SSSP Tree (Source: " << source << "):" << endl;
        cout << "Vertex | Distance | Parent" << endl;
        cout << "-------|----------|-------" << endl;
        int limit = dist.size() < static_cast<unsigned long>(maxVertices) ? dist.size() : maxVertices;
        for (int v = 0; v < limit; ++v) {
            cout << v << "      | ";
            if (dist[v] == numeric_limits<double>::infinity()) {
                cout << "INF      | ";
            } else {
                cout << dist[v] << "      | ";
            }
            cout << parent[v] << endl;
        }
        cout << "(Showing first " << limit << " vertices)" << endl;
    }
};

// Priority queue comparator for Dijkstra
struct Compare {
    bool operator()(const pair<double, int>& a, const pair<double, int>& b) {
        return a.first > b.first; // Min-heap
    }
};

// Compute initial SSSP using Dijkstra's algorithm
void computeInitialSSSP(const Graph& G, SSSPTree& T) {
    if (T.dist.size() < static_cast<unsigned long>(G.numVertices)) {
        T.resize(G.numVertices);
    }

    priority_queue<pair<double, int>, vector<pair<double, int> >, Compare> pq;
    unordered_set<int> visited;
    pq.push(pair<double, int>(0.0, T.source));

    while (!pq.empty()) {
        double d = pq.top().first;
        int u = pq.top().second;
        pq.pop();

        if (u >= G.numVertices) {
            cerr << "Warning: Skipping invalid vertex " << u << endl;
            continue;
        }
        if (visited.find(u) != visited.end()) continue;
        visited.insert(u);

        for (vector<pair<int, double> >::const_iterator neighbor = G.adj[u].begin(); neighbor != G.adj[u].end(); ++neighbor) {
            int v = neighbor->first;
            double w = neighbor->second;
            if (v >= G.numVertices) {
                cerr << "Warning: Skipping invalid neighbor " << v << endl;
                continue;
            }
            if (visited.find(v) == visited.end() && T.dist[v] > T.dist[u] + w) {
                T.dist[v] = T.dist[u] + w;
                T.parent[v] = u;
                pq.push(pair<double, int>(T.dist[v], v));
            }
        }
    }

    cout << "Reachable vertices: " << visited.size() << endl;
}

// Check if edge (u, v) is in the SSSP tree
bool isEdgeInSSSPTree(const SSSPTree& T, int u, int v) {
    return (T.parent[v] == u && T.dist[v] < numeric_limits<double>::infinity());
}

// Update SSSP for a single edge change (insertion or deletion)
void updateSSSP(Graph& G, SSSPTree& T, const Edge& e, bool isInsertion) {
    int updatedVertices = 0;

    // Resize T if necessary
    if (e.u >= static_cast<int>(T.dist.size()) || e.v >= static_cast<int>(T.dist.size())) {
        T.resize(e.u > e.v ? e.u + 1 : e.v + 1);
    }

    // Step 1: Identify affected vertex
    int x, y;
    if (T.dist[e.u] > T.dist[e.v]) {
        x = e.u;
        y = e.v;
    } else {
        x = e.v;
        y = e.u;
    }

    priority_queue<pair<double, int>, vector<pair<double, int> >, Compare> pq;
    unordered_set<int> affected;

    if (isInsertion) {
        G.addEdge(e.u, e.v, e.weight);
        if (G.numVertices > static_cast<int>(T.dist.size())) {
            T.resize(G.numVertices);
        }
        double newDist = T.dist[y] + e.weight;
        if (T.dist[x] > newDist) {
            T.dist[x] = newDist;
            T.parent[x] = y;
            affected.insert(x);
            pq.push(pair<double, int>(T.dist[x], x));
            updatedVertices++;
        }
    } else {
        if (isEdgeInSSSPTree(T, e.u, e.v)) {
            if (T.parent[x] == y) {
                T.dist[x] = numeric_limits<double>::infinity();
                T.parent[x] = -1;
                affected.insert(x);
                pq.push(pair<double, int>(T.dist[x], x));
                updatedVertices++;
            }
        }
        G.removeEdge(e.u, e.v);
    }

    // Step 2: Update affected subgraph
    while (!pq.empty()) {
        int z = pq.top().second;
        pq.pop();
        if (z >= G.numVertices || affected.find(z) == affected.end()) continue;
        affected.erase(z);

        for (vector<pair<int, double> >::const_iterator neighbor = G.adj[z].begin(); neighbor != G.adj[z].end(); ++neighbor) {
            int n = neighbor->first;
            double w = neighbor->second;
            if (n >= G.numVertices) continue;
            double newDist = (T.dist[z] == numeric_limits<double>::infinity()) 
                           ? numeric_limits<double>::infinity() 
                           : T.dist[z] + w;
            if (T.dist[n] > newDist) {
                T.dist[n] = newDist;
                T.parent[n] = z;
                if (affected.find(n) == affected.end()) {
                    affected.insert(n);
                    pq.push(pair<double, int>(T.dist[n], n));
                    updatedVertices++;
                }
            }
        }
    }

    //cout << (isInsertion ? "Inserting" : "Deleting") << " edge (" << e.u << ", " << e.v << ", " << e.weight << "):" << endl;
    //cout << "Updated " << updatedVertices << " vertices in SSSP update" << endl;
}

// Read graph from file, handling sparse vertex IDs and storing edge list
void readGraph(const string& filename, int& numVertices, long long& numEdges,
               unordered_map<int, int>& idMap, vector<pair<int, int>>& edgeList) {
    ifstream file(filename.c_str());
    if (!file.is_open()) {
        cerr << "Error: Could not open " << filename << endl;
        exit(1);
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
        }
    }
    file.close();

    if (vertices.empty()) {
        cerr << "Error: No valid vertices found in " << filename << endl;
        exit(1);
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

    cout << "Graph Statistics:" << endl;
    cout << "----------------" << endl;
    cout << "Vertices: " << numVertices << endl;
    cout << "Edges: " << edgesRead << endl;
}

// Initialize updates (100 random operations)
void initializeUpdates(vector<pair<Edge, bool>>& changes, int numVertices,
                      const vector<pair<int, int>>& edgeList) {
    changes.clear();
    changes.reserve(100);
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

        if (op == 0 && additions < targetChanges/3) { // Addition
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
        } else if (op == 1 && deletions < targetChanges/3 && !edgeList.empty()) { // Deletion
            int idx = rand() % edgeList.size();
            u = edgeList[idx].first;
            v = edgeList[idx].second;
            edgeKey = static_cast<long long>(u) * numVertices + v;
            if (existingEdges.find(edgeKey) != existingEdges.end()) {
                changes.push_back({Edge{u, v, 1.0}, false});
                existingEdges.erase(edgeKey);
                deletions++;
            }
        } else if (weightUpdates < targetChanges/3 && !edgeList.empty()) { // Weight update
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

    cout << "Generated " << changes.size() << " changes: " 
         << additions << " additions, " << deletions << " deletions, " 
         << weightUpdates << " weight updates" << endl;
}

int main(int argc, char* argv[]) {
    string graphFile = argc > 1 ? argv[1] : "data.txt";
    string partitionFile = "partition.txt";

    auto megaStart = chrono::high_resolution_clock::now();
    auto start = chrono::high_resolution_clock::now();
    auto end = chrono::high_resolution_clock::now();
    double graphLoadingTime = 0.0;
    double initialSSSPTime = 0.0;
    double totalUpdateTime = 0.0;

    // Time graph loading
    start = chrono::high_resolution_clock::now();

    int numVertices = 0;
    long long numEdges = 0;
    unordered_map<int, int> idMap;
    vector<pair<int, int>> edgeList;

    readGraph(graphFile, numVertices, numEdges, idMap, edgeList);

    end = chrono::high_resolution_clock::now();
    graphLoadingTime = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;

    // Initialize graph
    Graph G(numVertices);
    for (const auto& edge : edgeList) {
        G.addEdge(edge.first, edge.second, 1.0);
    }

    // Initialize SSSP tree
    SSSPTree T(G.numVertices, 0);

    // Time initial SSSP
    start = chrono::high_resolution_clock::now();

    computeInitialSSSP(G, T);

    end = chrono::high_resolution_clock::now();
    initialSSSPTime = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;

    // Initialize changes
    vector<pair<Edge, bool>> changes;
    initializeUpdates(changes, numVertices, edgeList);

    // Process updates with timing
    int changeIndex = 1;
    totalUpdateTime = 0.0;
    for (const auto& change : changes) {
        start = chrono::high_resolution_clock::now();

        updateSSSP(G, T, change.first, change.second);

        end = chrono::high_resolution_clock::now();
        totalUpdateTime += chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
        changeIndex++;
    }

    // Write distances
    ofstream out("distances.txt");
    for (int v = 0; v < numVertices; ++v) {
        if (T.dist[v] != numeric_limits<double>::infinity()) {
            out << v << " " << T.dist[v] << endl;
        }
    }
    out.close();

    // Final execution time summary
    auto megaEnd = chrono::high_resolution_clock::now();
    double totalExecutionTime = chrono::duration_cast<chrono::microseconds>(megaEnd - megaStart).count() / 1000.0;
    
    cout << "\nPerformance Summary:" << endl;
    cout << "------------------" << endl;
    cout << "Graph Loading:     " << graphLoadingTime << " ms" << endl;
    cout << "Initial SSSP:      " << initialSSSPTime << " ms" << endl;
    cout << "Total Updates:     " << totalUpdateTime << " ms" << endl;
    cout << "Avg Update Time:   " << totalUpdateTime / changes.size() << " ms" << endl;
    cout << "Total Runtime:     " << totalExecutionTime << " ms" << endl;
    cout << "------------------" << endl;

    return 0;
}