package maximum.industries


fun calcElo(s1: Double = 0.5, elo: Pair<Double,Double>, k: Double = 40.0): Pair<Double,Double> {
    val r1 = Math.pow(10.0,(elo.first/400))
    val r2 = Math.pow(10.0,(elo.second/400))

    val e1 = r1/(r1+r2)
    val e2 = r2/(r1+r2)

    val s2 = 1-s1

    val r1_ = r1 + k*(s1-e1)
    val r2_ = r2 + k*(s2-e2)

    return Pair(r1_,r2_)
}


data class Contestant( val strategy: GameSearchAlgo, var elo: Double = 1500.0)



fun runTournament(players: Array<Contestant>, game: GameGrammar.GameSpec) {
    for (player1 in players) {
        for (player2 in players) {
            if (player1==player2) continue
            val result = play(game,player1.strategy,player2.strategy,null)
            val newElo = calcElo(result, Pair(player1.elo,player2.elo))
            player1.elo = newElo.first
            player2.elo = newElo.second
        }
    }
}