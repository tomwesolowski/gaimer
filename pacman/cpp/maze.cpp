#include <bits/stdc++.h>
#include "pstream.h"

using namespace std;
#include "bot.h"

#define y first
#define x second

int main() {
  ios_base::sync_with_stdio(0);
  srand(time(0));

  redi::pstream proc("python3 -u ../maze.py");

  while(true) {
    // read_board
		vector<string> board;
		vector<vector<int>> intboard;
		string line;
		std::getline(proc.out(), line);
		stringstream ss;
		ss << line << endl;
		ss >> HEIGHT >> WIDTH;
		if(HEIGHT < 0 || WIDTH < 0) {
			// game is over.
			return 0;
		}
		for(int y = 0; y < HEIGHT; y++) {
			std::getline(proc.out(), line);
			board.push_back(line);
			intboard.push_back(vector<int>());
			for(char c : line) {
				intboard.back().push_back(c-'0');
			}
		}

		pair<int, int> exit;
		std::getline(proc.out(), line);
		ss << line << endl;
		ss >> exit.y >> exit.x;

		std::getline(proc.out(), line);
		int num_agents = stoi(line);
		vector< pair<int, int> > agents;
		for(int i = 0; i < num_agents; i++) {
			std::getline(proc.out(), line);
			ss << line << endl;
			int y, x;
			ss >> y >> x;
			agents.push_back({y, x});
		}

		std::getline(proc.out(), line);
		int num_foods = stoi(line);
		vector< pair<int, int> > foods;
		for(int i = 0; i < num_foods; i++) {
			std::getline(proc.out(), line);
			ss << line << endl;
			int y, x;
			ss >> y >> x;
			foods.push_back({y, x});
		}

		// make move.
		string m;
		m += '0'+play(intboard, agents, foods, exit);
		proc << m << endl;
		usleep(WAIT*1000);
  }
}