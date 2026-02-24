board = [
    [" ", " ", " "],
    [" ", " ", " "],
    [" ", " ", " "]
]

def display_board(board):
    print("TIC TAC TOE")
    print("*" * 17)
    for i, row in enumerate(board):
        print(f"*   {row[0]} | {row[1]} | {row[2]}   *")
        if i < 2:
            print("*  ---|---|---  *")
    print("*" * 17)

def player_input(player, board):
    while True:
        try:
            row = int(input(f"Player {player}, enter row (1-3): ")) - 1
            col = int(input(f"Player {player}, enter column (1-3): ")) - 1
            if row in range(3) and col in range(3) and board[row][col] == " ":
                return row, col
            else:
                print("Invalid move. Try again.")
        except ValueError:
            print("Please enter a number.")

def check_win(board, player):
    for row in board:
        if all(cell == player for cell in row):
            return True
    for col in range(3):
        if all(board[row][col] == player for row in range(3)):
            return True
    if all(board[i][i] == player for i in range(3)):
        return True
    if all(board[i][2 - i] == player for i in range(3)):
        return True
    return False

def check_tie(board):
    return all(board[row][col] != " " for row in range(3) for col in range(3))

def play():
    print("Welcome to TIC TAC TOE!")
    board = [
        [" ", " ", " "],
        [" ", " ", " "],
        [" ", " ", " "]
    ]
    current_player = "X"
    while True:
        display_board(board)
        print(f"Player {current_player}'s turn...")
        row, col = player_input(current_player, board)
        board[row][col] = current_player
        if check_win(board, current_player):
            display_board(board)
            print(f"Player {current_player} wins!")
            break
        if check_tie(board):
            display_board(board)
            print("It's a tie!")
            break
        if current_player == "X":
            current_player = "O"
        else:
            current_player = "X"

play()
