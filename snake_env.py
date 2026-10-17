import json
import random
from collections import deque

DIRECTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

OPPOSITES = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}

SNAKE_QUESTIONS = {
    "move": {
        "type": "choice",
        "instructions": "Which direction should the snake move next to eat the food without dying?",
        "criteria": {
            "up": "move up (y-1)",
            "down": "move down (y+1)",
            "left": "move left (x-1)",
            "right": "move right (x+1)",
        },
    },
    "danger": {
        "type": "noul",
        "instructions": "Is the snake in immediate danger of dying next move?",
    },
    "risk": {
        "type": "score",
        "instructions": "How risky is the current position?",
        "criteria": ["safe", "some risk", "high risk", "critical"],
    },
}


class SnakeEnv:
    def __init__(self, width=10, height=10, seed=42):
        self.width = int(width)
        self.height = int(height)
        self.seed = int(seed) if seed is not None else 42
        self.rng = random.Random(self.seed)
        self.reset(self.seed)

    def reset(self, seed=None):
        if seed is not None:
            self.seed = int(seed)
        self.rng = random.Random(self.seed)
        cx, cy = self.width // 2, self.height // 2
        # Serpente lungo 3, orientato verso destra
        self.snake = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])
        self.direction = "right"
        self.score = 0
        self.steps = 0
        self.alive = True
        self.death_cause = None  # "wall", "body", "illegal_180", "starvation", "won"
        self.food = self._spawn_food()
        return self

    def _spawn_food(self):
        occupied = set(self.snake)
        free_cells = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in occupied
        ]
        if not free_cells:
            return None
        return self.rng.choice(free_cells)

    def _in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_move_safe(self, d_name):
        if d_name not in DIRECTIONS:
            return False
        # Inversione a 180 gradi è illegale/mortale
        if d_name == OPPOSITES.get(self.direction):
            return False
        hx, hy = self.snake[0]
        dx, dy = DIRECTIONS[d_name]
        nx, ny = hx + dx, hy + dy
        if not self._in_bounds(nx, ny):
            return False
        # Se non mangia cibo, la coda si sposta liberando l'ultima cella
        will_eat = (self.food is not None and (nx, ny) == self.food)
        body_check = list(self.snake) if will_eat else list(self.snake)[:-1]
        if (nx, ny) in body_check:
            return False
        return True

    def _distance_to_obstacle(self, d_name):
        hx, hy = self.snake[0]
        dx, dy = DIRECTIONS[d_name]
        body_set = set(list(self.snake)[:-1])
        dist = 0
        cx, cy = hx, hy
        while True:
            cx += dx
            cy += dy
            dist += 1
            if not self._in_bounds(cx, cy) or (cx, cy) in body_set:
                return dist

    def _flood_fill_reachable(self, d_name):
        if not self.is_move_safe(d_name):
            return 0
        hx, hy = self.snake[0]
        dx, dy = DIRECTIONS[d_name]
        nx, ny = hx + dx, hy + dy
        will_eat = (self.food is not None and (nx, ny) == self.food)
        new_body = [ (nx, ny) ] + (list(self.snake) if will_eat else list(self.snake)[:-1])
        obstacles = set(new_body[1:])  # tutte le parti del corpo tranne la nuova testa

        visited = {(nx, ny)}
        queue = deque([(nx, ny)])
        while queue:
            cx, cy = queue.popleft()
            for mdx, mdy in DIRECTIONS.values():
                tx, ty = cx + mdx, cy + mdy
                if self._in_bounds(tx, ty) and (tx, ty) not in obstacles and (tx, ty) not in visited:
                    visited.add((tx, ty))
                    queue.append((tx, ty))
        return len(visited)

    def get_sensors(self):
        hx, hy = self.snake[0]
        fx, fy = self.food if self.food is not None else (hx, hy)
        curr_dist = abs(fx - hx) + abs(fy - hy)

        directions_info = {}
        for d_name, (dx, dy) in DIRECTIONS.items():
            nx, ny = hx + dx, hy + dy
            safe = self.is_move_safe(d_name)
            is_180 = (d_name == OPPOSITES.get(self.direction))
            new_dist = abs(fx - nx) + abs(fy - ny)
            moves_toward_food = bool(new_dist < curr_dist)
            dist_obs = 1 if is_180 else self._distance_to_obstacle(d_name)
            reachable = self._flood_fill_reachable(d_name)
            directions_info[d_name] = {
                "safe": safe,
                "is_illegal_180_reversal": is_180,
                "distance_to_obstacle": dist_obs,
                "moves_toward_food": moves_toward_food,
                "reachable_space": reachable,
            }
        return directions_info

    def get_ascii_grid(self):
        lines = ["#" * (self.width + 2)]
        body_set = set(list(self.snake)[1:])
        hx, hy = self.snake[0]
        for y in range(self.height):
            row = ["#"]
            for x in range(self.width):
                if (x, y) == (hx, hy):
                    row.append("H")
                elif (x, y) in body_set:
                    row.append("o")
                elif self.food is not None and (x, y) == self.food:
                    row.append("F")
                else:
                    row.append(".")
            row.append("#")
            lines.append("".join(row))
        lines.append("#" * (self.width + 2))
        return "\n".join(lines)

    def get_state(self, mode="sensors_and_grid"):
        """
        mode può essere:
        - "sensors_and_grid" (default: sensori + griglia)
        - "sensors_only" (solo sensori)
        - "grid_only" (solo griglia ASCII)
        """
        hx, hy = self.snake[0]
        fx, fy = self.food if self.food is not None else (hx, hy)

        if mode == "grid_only":
            return {
                "grid": self.get_ascii_grid().split("\n"),
                "current_direction": self.direction,
                "legend": "#=wall, H=head, o=body, F=food, .=empty",
            }

        sensors = self.get_sensors()
        state = {
            "head": {"x": hx, "y": hy},
            "current_direction": self.direction,
            "length": len(self.snake),
            "food": {"x": fx, "y": fy},
            "food_delta": {"dx": fx - hx, "dy": fy - hy},
            "directions": sensors,
        }
        if mode == "sensors_and_grid":
            state["grid"] = self.get_ascii_grid().split("\n")
        return state

    def get_safe_moves(self):
        return [d for d in DIRECTIONS if self.is_move_safe(d)]

    def step(self, action):
        if not self.alive:
            return {"ate": False, "alive": False, "death_cause": self.death_cause}

        self.steps += 1
        action = str(action).lower().strip()
        if action not in DIRECTIONS:
            action = self.direction

        # Controlla inversione a 180 gradi ("mossa illegale")
        if action == OPPOSITES.get(self.direction):
            self.alive = False
            self.death_cause = "illegal_180"
            return {"ate": False, "alive": False, "death_cause": self.death_cause}

        hx, hy = self.snake[0]
        dx, dy = DIRECTIONS[action]
        nx, ny = hx + dx, hy + dy

        # Collisione col muro
        if not self._in_bounds(nx, ny):
            self.direction = action
            self.alive = False
            self.death_cause = "wall"
            return {"ate": False, "alive": False, "death_cause": self.death_cause}

        will_eat = (self.food is not None and (nx, ny) == self.food)
        body_to_check = list(self.snake) if will_eat else list(self.snake)[:-1]

        # Collisione col corpo
        if (nx, ny) in body_to_check:
            self.direction = action
            self.alive = False
            self.death_cause = "body"
            return {"ate": False, "alive": False, "death_cause": self.death_cause}

        # Movimento valido
        self.direction = action
        self.snake.appendleft((nx, ny))
        if will_eat:
            self.score += 1
            self.food = self._spawn_food()
            if self.food is None:
                self.alive = False
                self.death_cause = "won"
        else:
            self.snake.pop()

        return {"ate": will_eat, "alive": self.alive, "death_cause": self.death_cause}


def policy_random_safe(env: SnakeEnv):
    """Agente casuale-sicuro: sceglie a caso tra le mosse sicure."""
    safe_moves = env.get_safe_moves()
    if not safe_moves:
        return env.direction
    return env.rng.choice(safe_moves)


def policy_greedy(env: SnakeEnv):
    """Agente greedy: sceglie la mossa sicura che avvicina al cibo (con tie-break su spazio raggiungibile)."""
    safe_moves = env.get_safe_moves()
    if not safe_moves:
        return env.direction
    hx, hy = env.snake[0]
    fx, fy = env.food if env.food is not None else (hx, hy)
    sensors = env.get_sensors()

    best_move = safe_moves[0]
    best_key = None
    for m in safe_moves:
        dx, dy = DIRECTIONS[m]
        nx, ny = hx + dx, hy + dy
        manhattan = abs(fx - nx) + abs(fy - ny)
        reach = sensors[m]["reachable_space"]
        # Minimizza distanza manhattan, massimizza reachable_space
        key = (-manhattan, reach)
        if best_key is None or key > best_key:
            best_key = key
            best_move = m
    return best_move
