package max.dillon
import java.awt.*
import javax.swing.JFrame


class GameWindow: JFrame() {

    init {
        setSize(640,480)
        title = "Game Window"
        defaultCloseOperation = JFrame.EXIT_ON_CLOSE
        isVisible = true

    }


}