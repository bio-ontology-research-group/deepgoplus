#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <map>
#include <set>
#include <bitset>
#include <pthread.h>
#include <fstream>
#include <climits>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <boost/threadpool.hpp>
#include <cstring>
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>

#define NUM_NODES 9000000
#define BUFFERSIZE 65536
#define THREADS 8

#define NUMBER_WALKS 100
#define LENGTH_WALKS 20

using namespace std;
using namespace boost::threadpool;


struct Edge {
    unsigned int edge ;
    unsigned int node ;
};

unordered_map<unsigned int, vector<Edge> > graph ;

random_device rd;
mt19937 rng(rd());
uniform_int_distribution<int> uni(0, INT_MAX);


ofstream fout;
boost::mutex mtx;


void build_graph(string fname) {
    graph.reserve(NUM_NODES) ;
    ifstream in(fname);
    string line;
    unsigned int source, node, edge;
    stringstream ss;
    while (getline(in, line)) {
	ss.clear();
	ss << line;
	Edge e;
	ss >> source >> node >> edge;
	e.node = node;
	e.edge = edge;
	graph[source].push_back(e);
    }
}

void walk(unsigned int source) {
    vector<vector<unsigned int> > walks(NUMBER_WALKS) ;
    if (graph[source].size() > 0) { // if there are outgoing edges at all
	for (int i = 0 ; i < NUMBER_WALKS ; i++) {
	    int count = LENGTH_WALKS ;
	    int current = source ;
	    walks[i].push_back(source) ;
	    while (count > 0) {
		if (graph[current].size() > 0 ) { // if there are outgoing edges
		    unsigned int r = uni(rng) % graph[current].size();
		    Edge next = graph[current][r] ;
		    int target = next.node ;
		    int edge = next.edge ;
		    // walks[i].push_back(edge) ;
		    walks[i].push_back(target) ;
		    current = target ;
		} else {
		    int edge = INT_MAX ; // null edge
		    current = source ;
		    // walks[i].push_back(edge) ;
		    walks[i].push_back(current) ;
		}
		count-- ;
	    }
	}
    }
    stringstream ss;
    for(vector<vector<unsigned int> >::iterator it = walks.begin(); it != walks.end(); ++it) {
	for(size_t i = 0; i < (*it).size(); ++i) {
	    if(i != 0) {
		ss << " ";
	    }
	    ss << (*it)[i];
	}
	ss << "\n" ;
    }
    mtx.lock() ;
    fout << ss.str() ;
    fout.flush() ;
    mtx.unlock() ;
}

void generate_corpus() {
    pool tp(THREADS);
    for ( auto it = graph.begin(); it != graph.end(); ++it ) {
	unsigned int source = it -> first ;
	tp.schedule(boost::bind(&walk, source ) ) ;
    }
    cout << tp.pending() << " tasks pending." << "\n" ;
    tp.wait() ;
}

int main (int argc, char *argv[]) {
    cout << "Building graph from " << argv[1] << "\n" ;
    build_graph(argv[1]);
    cout << "Number of nodes in graph: " << graph.size() << "\n" ;
    cout << "Writing walks to " << argv[2] << "\n" ;
    fout.open(argv[2]) ;
    generate_corpus() ;
    fout.close() ;
}
