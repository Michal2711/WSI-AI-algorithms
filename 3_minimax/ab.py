from copy import deepcopy


def alfabeta(game, depth, max_player, alfa, beta):
    if depth == 0 or len(game.get_moves()) == 0:
        return game.state.evaluate(), game

    if max_player.char == game.second_player.char:
        best_move = None
        for move in game.get_moves():
            temp_game = deepcopy(game)
            temp_game.make_move(move)
            evaluation = alfabeta(temp_game, depth-1, game.first_player, alfa, beta)
            if alfa < evaluation[0]:
                alfa = evaluation[0]
                best_move = move
            if alfa >= beta:
                break
        return alfa, best_move

    elif max_player.char == game.first_player.char:
        best_move = None
        for move in game.get_moves():
            temp_game = deepcopy(game)
            temp_game.make_move(move)
            evaluation = alfabeta(temp_game, depth-1, game.second_player, alfa, beta)
            if beta > evaluation[0]:
                beta = evaluation[0]
                best_move = move
            if alfa >= beta:
                break

        return beta, best_move
