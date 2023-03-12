from copy import deepcopy


def minimax(game, depth, max_player):
    if depth == 0 or len(game.get_moves()) == 0:
        return game.state.evaluate(), game

    if max_player.char == game.second_player.char:
        maxEval = float('-inf')
        best_move = None
        for move in game.get_moves():
            temp_game = deepcopy(game)
            temp_game.make_move(move)
            evaluation = minimax(temp_game, depth-1, game.first_player)

            if maxEval < evaluation[0]:
                maxEval = evaluation[0]
                best_move = move

        return maxEval, best_move

    elif max_player.char == game.first_player.char:
        minEval = float('inf')
        best_move = None

        for move in game.get_moves():
            temp_game = deepcopy(game)
            temp_game.make_move(move)
            evaluation = minimax(temp_game, depth-1, game.second_player)

            if minEval > evaluation[0]:
                minEval = evaluation[0]
                best_move = move

        return minEval, best_move
