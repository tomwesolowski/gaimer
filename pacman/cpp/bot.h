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

typedef pair<int, int> point;

const int inf = 1e9;

#define y first
#define x second

int HEIGHT, WIDTH;

vector< pair<int, int> > dirs = {
	{0, 0}, // IDLE
	{-1, 0}, // ^
	{0, 1}, // >
	{1, 0}, // V
	{0, -1} // <
};

point move(point p, int dir) {
	return {p.y + dirs[dir].y, p.x + dirs[dir].x};
}

bool good(point p) {
	return p.y >= 0 && p.x >= 0 && p.y < HEIGHT && p.x < WIDTH;
}

bool free(const vector<vector<int>>& board, point p) {
	return good(p) && board[p.y][p.x] != WALL;
}

int revdir(int dir) {
	if(dir == IDLE) {
		return IDLE;
	}
	dir--;
	dir += 2;
	dir %= 4;
	return dir + 1;
}

int get_move(const vector<vector<int>>& board, point start, vector<point> targets) {
	for(point target : targets) {
		if(start == target) {
			return IDLE;
		}	
	}
	queue<point> que;
	vector< vector<bool> > visited(HEIGHT, vector<bool>(WIDTH, false));
	for(point target : targets) {
		que.push(target);
		visited[target.y][target.x] = true;
	}
	while(que.size()) {
		point p = que.front(); que.pop();
		for(int dir = 0; dir < dirs.size(); dir++) {
			point q = move(p, dir);
			if(free(board, q)) {
				if(!visited[q.y][q.x]) {
					if(q == start) {
						return revdir(dir);
					}
					que.push(q);
					visited[q.y][q.x] = true;
				}
			}
		}
	}
	assert(false);
}

int get_dist(const vector<vector<int>>& board, point start, vector<point> targets) {
	for(point target : targets) {
		if(start == target) {
			return 0;
		}	
	}
	queue<point> que;
	vector< vector<bool> > visited(HEIGHT, vector<bool>(WIDTH, false));
	vector< vector<int> > dist(HEIGHT, vector<int>(WIDTH, inf));
	for(point target : targets) {
		que.push(target);
		visited[target.y][target.x] = true;
		dist[target.y][target.x] = 0;
	}
	while(que.size()) {
		point p = que.front(); que.pop();
		for(int dir = 0; dir < dirs.size(); dir++) {
			point q = move(p, dir);
			if(free(board, q)) {
				if(!visited[q.y][q.x]) {
					if(q == start) {
						return dist[p.y][p.x] + 1;
					}
					que.push(q);
					visited[q.y][q.x] = true;
					dist[q.y][q.x] = dist[p.y][p.x] + 1;
				}
			}
		}
	}
	assert(false);
}

const int IDLE_MODE = 0;
const int HUNGRY_MODE = 1;
const int FINAL_MODE = 2;
const int PANIC_MODE = 3;

const int CLOSE_THRESHOLD = 4;

int play(vector<vector<int>> board, 
				 vector< point > agents,
				 vector< point > foods,
				 point exit) {


	point player = agents.front();
	agents.erase(agents.begin());

	// wybierz mode
	int mode = IDLE_MODE;

	if(get_dist(board, player, agents) <= CLOSE_THRESHOLD) {
		mode = PANIC_MODE;
	}
	else if(foods.size() > 0) {
		mode = HUNGRY_MODE;
	}
	else {
		mode = FINAL_MODE;
	}

	cerr << "MODE: " << mode << endl;

	if(mode == HUNGRY_MODE) {
		return get_move(board, player, foods);
	}
	else if(mode == FINAL_MODE) {
		return get_move(board, player, {exit});
	}
	else if(mode == PANIC_MODE) {
		int best_move = IDLE;
		double best_move_dist = -inf;
		for(int dir = 0; dir < dirs.size(); dir++) {
			point q = move(player, dir);
			if(!free(board, q)) continue;
			double qdist = get_dist(board, q, agents);
			if(foods.size()) {
				qdist -= (double)get_dist(board, q, foods) / 10;
			}
			else {
				qdist -= (double)get_dist(board, q, {exit}) / 10;
			}
			if(qdist > best_move_dist) {
				best_move_dist = qdist;
				best_move = dir;
			}
		}
		return best_move;
	}
	else {
		return IDLE;
	}
}
