from algorithm import minimax
from dots_and_boxes import DotsAndBoxes, DotsAndBoxesMove
from player import Player
from ab import alfabeta


def main():
    game = DotsAndBoxes(2, Player('A'), Player('B'))
    print(game)
    while game.is_finished() is False:
        if game.state.current_player == game.second_player:
            value, move = alfabeta(game, 4, game.second_player, float('-inf'), float('inf'))
            # value, move = minimax(game, 4, game.second_player)
            game.make_move(move)
        else:
            moves = game.get_moves()
            for index, move in enumerate(moves, start=0):
                print(f'{index}. {move.connection}, {move.loc}')
            choice = int(input("Enter a choice: "))
            move = moves[choice]
            game.make_move(move)
        print(game.get_scores())
        print(game)
    print(f' THE WINNER IS {game.get_winner()}')


if __name__ == "__main__":
    main()
