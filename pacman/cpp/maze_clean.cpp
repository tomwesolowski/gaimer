#include <bits/stdc++.h>
#include "pstream.h"

using namespace std;

const int EMPTY = 0;
const int WALL = 1;
const int EXIT = 2;
const int PLAYER = 3;
const int GHOST = 4;

const int IDLE = 0;
const int UP = 1;
const int RIGHT = 2;
const int DOWN = 3;
const int LEFT = 4;
const int NUM_MOVES = 5;

const int WAIT = 500;

int dy[] = {0, -1, 0, 1, 0};
int dx[] = {0, 0, 1, 0, -1};

int play(const vector<vector<int>>& board, 
				 const vector< pair<int, int> >& agents,
				 const vector< pair<int, int> >& foods) {
	int width = board.front().size();
	int height = board.size();

	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			if(board[y][x] == PLAYER) {
				cerr << y << " " << x << endl;
				break;
			}
		}
	}

	return rand()%NUM_MOVES;
}

int main() {
  ios_base::sync_with_stdio(0);
  srand(time(0));

  redi::pstream proc("python3 -u ./maze.py");

  while(true) {
    // read_board
		vector<string> board;
		vector<vector<int>> intboard;
		string line;
		std::getline(proc.out(), line);
		stringstream ss(line);
		int height, width;
		ss >> height >> width;
		if(height < 0 || width < 0) {
			// game is over.
			return 0;
		}
		for(int y = 0; y < height; y++) {
			std::getline(proc.out(), line);
			board.push_back(line);
			intboard.push_back(vector<int>());
			for(char c : line) {
				intboard.back().push_back(c-'0');
			}
		}

		std::getline(proc.out(), line);
		cerr << "Agents: " << line << endl;
		int num_agents = stoi(line);
		vector< pair<int, int> > agents;
		for(int i = 0; i < num_agents; i++) {
			std::getline(proc.out(), line);
			ss << line;
			int y, x;
			ss >> y >> x;
			agents.push_back({y, x});
		}

		std::getline(proc.out(), line);
		cerr << "Foods: " << line << endl;
		int num_foods = stoi(line);
		vector< pair<int, int> > foods;
		for(int i = 0; i < num_foods; i++) {
			std::getline(proc.out(), line);
			ss << line;
			int y, x;
			ss >> y >> x;
			foods.push_back({y, x});
		}

		// make move.
		string m;
		m += '0'+play(intboard, agents, foods);
		cerr << m << endl;
		proc << m << endl;
		usleep(WAIT*1000);
  }
}