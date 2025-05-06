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
Graph readGraph(const string& filename, int& expectedNodes, long long& expectedEdges, 
                unordered_map<int, int>& idMap, vector<pair<int, int> >& edgeList) {
    ifstream file(filename.c_str());
    if (!file.is_open()) {
        cerr << "Error: Could not open " << filename << endl;
        exit(1);
    }

    string line;
    long long edgesRead = 0;
    unordered_set<int> vertices;
    edgeList.clear();

    // First pass: Collect unique vertices and edges
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
            edgeList.push_back(pair<int, int>(u, v));
            edgesRead++;
            if (edgesRead % 1000000 == 0) {
                cout << "Scanned " << edgesRead << " edges..." << endl;
            }
        } else {
            cerr << "Warning: Skipping malformed line: " << line << endl;
        }
    }
    file.close();

    if (vertices.empty()) {
        cerr << "Error: No valid vertices found in " << filename << endl;
        exit(1);
    }

    // Create contiguous vertex IDs
    int numVertices = vertices.size();
    int newId = 0;
    for (unordered_set<int>::iterator v = vertices.begin(); v != vertices.end(); ++v) {
        idMap[*v] = newId++;
    }

    // Initialize graph with actual number of vertices
    Graph G(numVertices);
    cout << "Detected " << numVertices << " nodes (" << vertices.size() << " unique vertices)" << endl;
    cout << "Read " << edgesRead << " edges in first pass" << endl;
    if (expectedNodes > 0 && numVertices != expectedNodes) {
        cout << "Note: Detected nodes (" << numVertices << ") differ from header nodes (" << expectedNodes << "). Using unique vertex count." << endl;
    }
    if (expectedEdges > 0 && edgesRead != expectedEdges) {
        cout << "Warning: Expected " << expectedEdges << " edges, read " << edgesRead << endl;
    }

    // Second pass: Add edges with remapped IDs
    for (vector<pair<int, int> >::iterator edge = edgeList.begin(); edge != edgeList.end(); ++edge) {
        int u = idMap[edge->first];
        int v = idMap[edge->second];
        G.addEdge(u, v, 1.0);
    }

    return G;
}

int main() {
    string filename = "data.txt";
    cout << "Reading graph from " << filename << "..." << endl;

    // Time graph loading
    auto megastart = chrono::high_resolution_clock::now();
    auto start = chrono::high_resolution_clock::now();
    int expectedNodes = 0;
    long long expectedEdges = 0;
    unordered_map<int, int> idMap;
    vector<pair<int, int> > edgeList;
    Graph G = readGraph(filename, expectedNodes, expectedEdges, idMap, edgeList);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
    double graphLoadingTime = duration / 1000.0;
    cout << "Graph loading time: " << graphLoadingTime << " ms" << endl;

    cout << "\nInitial Graph:" << endl;
    G.printGraphSummary();

    SSSPTree T(G.numVertices, 0);
    cout << "\nComputing Initial SSSP:" << endl;
    // Time initial SSSP
    start = chrono::high_resolution_clock::now();
    computeInitialSSSP(G, T);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
    double initialSSSPTime = duration / 1000.0;
    cout << "Initial SSSP computation time: " << initialSSSPTime << " ms" << endl;
    T.printTree();

    // Initialize random number generator
    srand(time(0));

    // Dynamic changes: 100,000 operations
    vector<pair<Edge, bool> > changes;
    changes.reserve(100);
    unordered_set<long long> existingEdges; // Track edges to avoid duplicates (u,v as u*numVertices+v)
    for (vector<pair<int, int> >::iterator edge = edgeList.begin(); edge != edgeList.end(); ++edge) {
        int u = idMap[edge->first];
        int v = idMap[edge->second];
        existingEdges.insert(static_cast<long long>(u) * G.numVertices + v);
    }

    int targetChanges = 100;
    int additions = 0, deletions = 0, weightUpdates = 0;
    vector<int> vertices;
    for (unordered_map<int, int>::iterator it = idMap.begin(); it != idMap.end(); ++it) {
        vertices.push_back(it->second);
    }

    for (int i = 0; i < targetChanges; ++i) {
        // Choose operation: 0=addition, 1=deletion, 2=weight update
        int op = rand() % 3;
        int u, v;
        double w = (rand() % 95 + 5) / 10.0; // Random weight 0.5 to 10.0
        long long edgeKey;

        if (op == 0 && additions < 33) { // Addition
            do {
                u = vertices[rand() % vertices.size()];
                v = vertices[rand() % vertices.size()];
                if (u != v) { // Avoid self-loops
                    edgeKey = static_cast<long long>(u) * G.numVertices + v;
                }
            } while (u == v || existingEdges.find(edgeKey) != existingEdges.end());
            changes.push_back(pair<Edge, bool>(Edge(u, v, w), true));
            existingEdges.insert(edgeKey);
            additions++;
        } else if (op == 1 && deletions < 33 && !edgeList.empty()) { // Deletion
            int idx = rand() % edgeList.size();
            u = idMap[edgeList[idx].first];
            v = idMap[edgeList[idx].second];
            edgeKey = static_cast<long long>(u) * G.numVertices + v;
            if (existingEdges.find(edgeKey) != existingEdges.end()) {
                changes.push_back(pair<Edge, bool>(Edge(u, v, 1.0), false));
                existingEdges.erase(edgeKey);
                deletions++;
            }
        } else if (weightUpdates < 33 && !edgeList.empty()) { // Weight update (delete + add)
            int idx = rand() % edgeList.size();
            u = idMap[edgeList[idx].first];
            v = idMap[edgeList[idx].second];
            edgeKey = static_cast<long long>(u) * G.numVertices + v;
            if (existingEdges.find(edgeKey) != existingEdges.end()) {
                changes.push_back(pair<Edge, bool>(Edge(u, v, 1.0), false)); // Delete
                existingEdges.erase(edgeKey);
                changes.push_back(pair<Edge, bool>(Edge(u, v, w), true));   // Add with new weight
                existingEdges.insert(edgeKey);
                weightUpdates += 2; // Counts as two changes
                i++; // Skip next iteration since we added two changes
            }
        } else {
            // Fallback to addition if deletion/weight update not possible
            do {
                u = vertices[rand() % vertices.size()];
                v = vertices[rand() % vertices.size()];
                if (u != v) {
                    edgeKey = static_cast<long long>(u) * G.numVertices + v;
                }
            } while (u == v || existingEdges.find(edgeKey) != existingEdges.end());
            changes.push_back(pair<Edge, bool>(Edge(u, v, w), true));
            existingEdges.insert(edgeKey);
            additions++;
        }

        // For small graphs, break early if no more valid operations
        if (G.numVertices <= 6 && existingEdges.size() >= static_cast<unsigned long>(G.numVertices * (G.numVertices - 1))) {
            break;
        }
    }

    // Adjust changes to exactly 100,000 if needed
    while (changes.size() < 100000 && additions < 33333) {
        int u, v;
        long long edgeKey;
        double w = (rand() % 95 + 5) / 10.0;
        do {
            u = vertices[rand() % vertices.size()];
            v = vertices[rand() % vertices.size()];
            if (u != v) {
                edgeKey = static_cast<long long>(u) * G.numVertices + v;
            }
        } while (u == v || existingEdges.find(edgeKey) != existingEdges.end());
        changes.push_back(pair<Edge, bool>(Edge(u, v, w), true));
        existingEdges.insert(edgeKey);
        additions++;
    }

    cout << "\nApplying Dynamic Operations (" << changes.size() << " changes: " 
         << additions << " additions, " << deletions << " deletions, " << weightUpdates << " weight updates):" << endl;
    int changeIndex = 1;
    double totalUpdateTime = 0.0;
    for (vector<pair<Edge, bool> >::iterator change = changes.begin(); change != changes.end(); ++change) {
        // Time individual update
        auto startUpdate = chrono::high_resolution_clock::now();
        updateSSSP(G, T, change->first, change->second);
        auto endUpdate = chrono::high_resolution_clock::now();
        auto updateDuration = chrono::duration_cast<chrono::microseconds>(endUpdate - startUpdate).count();
        totalUpdateTime += updateDuration / 1000.0;

        if (changeIndex % 10000 == 0 || changeIndex == 1 || changeIndex == changes.size()) {
            cout << "\n---------------------------------" << endl;
            cout << "Change " << changeIndex << ":" << endl;
            cout << "Update time: " << updateDuration / 1000.0 << " ms" << endl;
            cout << "\nSSSP Tree After Update:" << endl;
            T.printTree();
            cout << "\nGraph After Update:" << endl;
            G.printGraphSummary();
        }
        changeIndex++;
    }
    auto megaend = chrono::high_resolution_clock::now();
    auto megatime = chrono::duration_cast<chrono::microseconds>(megaend - megastart).count();
    cout << "\nDynamic Updates Summary:" << endl;
    cout << "Total update time: " << totalUpdateTime << " ms" << endl;
    cout << "Average update time: " << totalUpdateTime / changes.size() << " ms" << endl;

    cout << "\nFinal Graph:" << endl;
    G.printGraphSummary();

    // Final execution time summary
    cout << "\nFinal Execution Time Summary:" << endl;
    cout << "-----------------------------" << endl;
    cout << "Graph loading time:          " << graphLoadingTime << " ms" << endl;
    cout << "Initial SSSP computation:    " << initialSSSPTime << " ms" << endl;
    cout << "Total dynamic updates:       " << totalUpdateTime << " ms" << endl;
    cout << "Average dynamic update:      " << totalUpdateTime / changes.size() << " ms" << endl;
    cout<<"Total time of execution is "<<megatime<<" s"<<endl;

    return 0;
}
