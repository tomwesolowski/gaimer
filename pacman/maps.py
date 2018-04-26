from pacman.pacman import *
from strategies import ChaseStrategy, PolicyStrategy, KeyboardStrategy
from approximators import LinearApproximator, TableLookupApproximator

'''
Simple Pacman Maps
'''


def get_very_simple_environment():
    board = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ]
    )

    params = PacmanParameters()
    return PacmanEnvironment(
        params=params,
        board=board,
        agents=[PlayerAgent("Player",
                            PolicyStrategy(approximator=TableLookupApproximator(params)),
                            Coord(0, 2)),
                GhostAgent("Ghost #1", ChaseStrategy(target_agent=PacmanAgentType.PLAYER), Coord(0, 6)),
                # GhostAgent("Ghost #2", ChaseStrategy(target_agent=AgentType.PLAYER), Coord(8, 10)),
                ],
        foods=[Coord(4, 6)],
        exit=Coord(0, 0))


def get_simple_environment_with_three_foods():
    board = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    params = PacmanParameters()
    return PacmanEnvironment(
        params=params,
        board=board,
        agents=[PlayerAgent("Player",
                            PolicyStrategy(approximator=LinearApproximator(PacmanFeatureExtractor())),
                            Coord(9, 0)),
                #GhostAgent("Ghost #1", ChaseStrategy(target_agent=AgentType.PLAYER), Coord(0, 8)),
                #GhostAgent("Ghost #2", ChaseStrategy(target_agent=AgentType.PLAYER), Coord(8, 10)),
                ],
        foods=[Coord(0, 10), Coord(4, 5), Coord(0, 0)],
        exit=Coord(10, 10))


def get_simple_environment_without_ghosts():
    board = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    exitposition = Coord(4, 8)
    params = PacmanParameters()
    return PacmanEnvironment(
        params=params,
        board=board,
        agents=[PlayerAgent("Player", PolicyStrategy(params=params), Coord(9, 0)),
                #GhostAgent("Ghost #1", ChaseStrategy(target_agent=AgentType.PLAYER), Coord(0, 8)),
                #GhostAgent("Ghost #2", ChaseStrategy(target_agent=AgentType.GHOST), Coord(8, 10)),
                #GhostAgent("Ghost #3", ChaseStrategy(target_pos=exitposition), Coord(4, 4))
                ],
        foods=[Coord(2, 0), Coord(0, 1), Coord(0, 2), Coord(1, 0), Coord(0, 0),
               Coord(3, 0), Coord(4, 0), Coord(10, 0), Coord(10, 1), Coord(10, 2),
               Coord(9, 2), Coord(10, 4), Coord(10, 5), Coord(10, 6), Coord(10, 7),
               Coord(0, 10), Coord(1, 10), Coord(2, 10), Coord(3, 10), Coord(4, 10),
               Coord(4, 4), Coord(4, 5), Coord(4, 6)],
        exit=exitposition)


def get_simple_environment():
    board = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    params = PacmanParameters()
    return PacmanEnvironment(
        params=params,
        board=board,
        agents=[PlayerAgent("Player", PolicyStrategy(approximator=TableLookupApproximator(params)), Coord(0, 0)),
                GhostAgent("Ghost #1", ChaseStrategy(target_agent=PacmanAgentType.PLAYER), Coord(10, 0)),
                GhostAgent("Ghost #2", ChaseStrategy(target_agent=PacmanAgentType.GHOST), Coord(10, 10)),
                ],
        foods=[Coord(2, 2), Coord(8, 2), Coord(0, 10)],
        exit=Coord(5, 4))