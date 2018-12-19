 

var formatBoard = function(board,side) {
    var p_map = [0,6,4,3,5,2,1,5,1]

    var newBoard = []
    board = board.map(x => p_map[Math.abs(x)]*Math.sign(x))
    for (var i = 0; i < board.length; i++) {
        newBoard[i] = {
            index: i+(board.length-2*i-1)*side,
            hilight: false,
            posibility: false,
            value: board[i+(board.length-2*i-1)*side]
        }
    }
    return newBoard 
}
var reloadBoard = function(board) {
    for (var i = 0; i < board.length; i++) {
        board[i].hilight = false
        board[i].posibility = false
    }
}

var app = angular.module('boardApp', []);
app.controller('myCtrl', function($scope, $http) {

    $scope.resetBoard = function() {
        $scope.whiteSide = !$scope.whiteSide
        $scope.sign = Math.sign(($scope.whiteSide?1:0)*2-1)
        $scope.startGame()
        
    }

    $scope.whiteSide = true;
    $scope.sign = Math.sign(($scope.whiteSide?1:0)*2-1)
    $scope.boardPixels = 640;
    $scope.selected = null;

    $scope.getColor = function(num) { return (Math.sign(-num.value)+1)/2 };
    $scope.getPiece = function(num) { return Math.abs(num.value)-1 };
    $scope.getRow = function(index) { return (index+Math.floor(index/$scope.size))%2 };

    $scope.modifyPossibilityIndex = function(index) {
        var x = index % 8
        var y = Math.floor(index / 8)
        var y_ = -(y + 1) % 8
        return 8 * y + x
    }
    $scope.updateBoard = function(data) {
        $scope.currentObject = data
        $scope.boardState = formatBoard(data.state.board, $scope.whiteSide)
        $scope.currentSignature = data.signature

        $scope.currentMoves = data.state.moves
        $scope.currentTurn = data.whiteMove
    };

    $scope.listPossibilities = function(piece) {
        for (var i = 0; i < $scope.currentMoves.length; i+=2) {
            var start = $scope.currentMoves[i]
            var end = $scope.currentMoves[i+1]
            if ($scope.whiteSide) {end = 63-end}
            if(piece.index == start) {
                $scope.boardState[end].posibility = true
            }
        }
    }

    $scope.startGame = function() {
        $http.post(
            "/start",
            $scope.whiteSide,
            { headers: {"Content-Type" : "application/json"}})
    .then(function(data) {
            $scope.updateBoard(data.data.first)
            $scope.gameSpec = data.data.second
            $scope.gameName = $scope.gameSpec.name_
            $scope.size = $scope.gameSpec.boardSize_
            console.log($scope.whiteSide)
            if (!$scope.whiteSide) { $scope.makeOpponentMove() }
        });
    }

    $scope.startGame()

    $scope.backgroundColor = function(piece,index) {
        if (piece.hilight) {
            if($scope.getRow(index)) {return "#6C6E47";} else { return "#8B966E";}
        } else if (piece.posibility){
            if($scope.getRow(index)) {return "#86756E";} else { return "#B5B2AD";}
        } else {
            if($scope.getRow(index)) {return "#B1886A";} else { return "#EFD8B9";}

        }
    }

    $scope.findMoveIndex = function(a, b) {
        for (var i = 0; i < $scope.currentMoves.length; i+=2) {
            if( $scope.currentMoves[i] == a &&
                $scope.currentMoves[i+1] == b) {
                return i / 2
            }
        }
        return -1
    }

    $scope.fail = function() {
            $scope.selected.hilight = false
            $scope.selected = null
    }

    $scope.makeOpponentMove = function() {
        $http.get("/opponentMove")
        .then(function(data){
            $scope.updateBoard(data.data)
            if ($scope.selected!=null) {$scope.selected.hilight = false}
            $scope.selected = null
        },function(data,status,header,config){
            $scope.fail()
        })
    }

    $scope.boardStateFromServer = function(squareFrom, squareTo) {
        $scope.currentObject.moveIndex = $scope.findMoveIndex(squareFrom.index, squareTo.index)
        if ($scope.currentObject.moveIndex==-1) {
            return $scope.fail()
        }
        $http.post(
            "/move",
            $scope.currentObject,
            {headers: {"Content-Type" : "application/json"} })
        .then(function(data,status,header,config ) {

            $scope.updateBoard(data.data)
            $scope.selected.hilight = false
            $scope.selected = null
            $scope.makeOpponentMove()

        },function(data,status,header,config){
            $scope.fail()
        });
    }

    $scope.selectPiece = function(piece) {
        reloadBoard($scope.boardState)

        if($scope.selected==piece) {
            if($scope.selected!=null) {$scope.selected.hilight=false;}
            $scope.selected=null
        } else if($scope.selected==null) {
            if (piece.value==0 || $scope.sign != Math.sign(piece.value)) {return}
            $scope.selected = piece
            $scope.listPossibilities(piece)
            $scope.selected.hilight = true
        } else {
            $scope.boardStateFromServer($scope.selected, piece)
        }

    }
});

