import constants

class TicTacToeEnv:
      

    def start(self):
        self.board = self.createBoard()
        self.turn = (0,1)
        return self.board

    def createBoard(self):
        board = []
        for y in range(3):
            row = []
            for x in range(3):
                row.append((0,0))
            board.append(row)
        return board


    def getLegalMoves(self, state):
        moves = []
        for x, row in enumerate(state):
            for y, square in enumerate(row):
                square = tuple(square)
                if square == (0, 0):
                    moves.append((y, x))

        return moves


    def move(self, move):
        move = constants.ACTION_LIST[move]
        self.board[move[1]][move[0]] = self.turn

        if self.turn == (0,1):
            self.turn = (1,0)
        else:
            self.turn = (0,1)

        done, winner = self.gameEnd()

        if done:
            if winner == (0, 0):
                moveValue = constants.DRAW_REWARD
            else:
                moveValue = constants.WIN_REWARD
        else:
            moveValue = 0

        return self.board, moveValue, done

    def gameEnd(self):


        ### on row ###
        toCheck = ()
        for row in self.board:
            toCheck = row[0]
            if row[1] == toCheck and row[2] == toCheck:
                if toCheck == (0,0):
                    continue
                return True, toCheck


        ### on column ###
        flipedBoard = []

        for x in range(3):
            subList = []
            for i in self.board:
                subList.append(i[x])
            flipedBoard.append(subList)


        toCheck = ()
        for row in flipedBoard:
            toCheck = row[0]
            if row[1] == toCheck and row[2] == toCheck:
                if toCheck == (0,0):
                    continue
                return True, toCheck



        ### on diagnal ###
        toCheck = self.board[0][0]
        if self.board[1][1] == toCheck and self.board[2][2] == toCheck:
            if toCheck != (0,0):
                return True, toCheck

        toCheck = self.board[0][2]
        if self.board[1][1] == toCheck and self.board[2][0] == toCheck:
            if toCheck != (0,0):
                return True, toCheck
        

        if 0 == len(self.getLegalMoves(self.board)):
            return True, (0,0)

        return False, (0,0)


    def getMask(self, state=None):
        if state == None:
            state = self.board
        legalMoves = self.getLegalMoves(state)
        mask = []
        for move in constants.ACTION_LIST:                                     
            if not move in legalMoves:
                mask.append(0)
            else:
                mask.append(1)
        return mask


    def isLegal(move):
        if self.getMask()[move] == 1:
            return  True
        return False

    def __repr__(self):
        string = ""
        for i in self.board:
            for x in i:
                if x == (0,1):
                    string += 'x '
                elif x == (1,0):
                    string += '0 ' 
                else:
                    string += '- ' 
            string += '\n'
        return string