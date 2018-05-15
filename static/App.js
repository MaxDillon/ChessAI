 


var formatBoard = function(board,side) {
    var newBoard = []
    for (var i = 0; i < board.length; i++) {
        newBoard[i] = {
            index: i+(board.length-2*i-1)*side,
            hilight: false,
            value: board[i+(board.length-2*i-1)*side]
        }
    }
    return newBoard 
}

var app = angular.module('myApp', []);
    app.controller('myCtrl', function($scope, $http) {
        $scope.whiteSide = true;

        $scope.getColor = function(num) { return (Math.sign(-num.value)+1)/2 };
        $scope.getPiece = function(num) { return Math.abs(num.value)-1 };
        $scope.getRow = function(index) { return (index+Math.floor(index/$scope.size))%2 };
        $scope.State = function(data) {
            this.board = formatBoard( data.board );
            this.moves = data.moves;
            this.whiteMove = data.whiteMove;
            this.moveDepth = data.moveDepth; 
        };
        $scope.updateBoard = function(data) { 

            $scope.currentObject = data
            $scope.boardState = formatBoard(data.state.board,$scope.whiteSide)
            $scope.currentSignature = data.signature

            $scope.currentMoves = data.state.moves
            $scope.currentTurn = data.whiteMove

        }; 

        $http.post("http://localhost:8080/start",$scope.whiteSide,{ headers: {"Content-Type" : "application/json"}}).then(function(data){
            $scope.updateBoard(data.data)
            $scope.size = Math.sqrt($scope.boardState.length)
        });
        $scope.findMoveIndex = function(a, b) {

            for (var i = 0; i < $scope.currentMoves.length; i+=2) {
                
                if( $scope.currentMoves[i]==a &&
                    $scope.currentMoves[i+1]==b) {
                    return i/2
                }
            }

            return -1
        }
        $scope.selected = null;
        // $scope.size = 8;

        $scope.fail = function() {
                $scope.selected.hilight = false
                $scope.selected = null
        }

        $scope.boardStateFromServer = function(squareFrom, squareTo) {
            
            $scope.currentObject.moveIndex = $scope.findMoveIndex(squareFrom.index, squareTo.index)
            if ($scope.currentObject.moveIndex==-1) {
                return $scope.fail()
            }
            $http.post("http://localhost:8080/move", $scope.currentObject, {headers: {"Content-Type" : "application/json"} })
            .then(function(data,status,header,config ) {

                $scope.updateBoard(data.data)
                $scope.selected.hilight = false
                $scope.selected = null
                
            },function(data,status,header,config){
                $scope.fail()
                // squareTo.value = squareFrom.value
                // squareTo.hilight = false
                // squareFrom.value = 0 
             });
        }




        

        $scope.selectPiece = function(piece) {   

            if($scope.selected==piece) { 
                if($scope.selected!=null) {$scope.selected.hilight=false;} 
                $scope.selected=null 
            } else if($scope.selected==null) {
                if (piece.value==0) {return}
                $scope.selected = piece
                $scope.selected.hilight = true
            } else {
                $scope.boardStateFromServer($scope.selected, piece)
            }
        }




    });

