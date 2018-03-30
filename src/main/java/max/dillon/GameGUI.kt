package max.dillon

import javax.swing.JButton
import javax.swing.plaf.basic.BasicButtonUI
import javax.swing.text.StyleConstants.setBackground
import javax.swing.AbstractButton
import javax.swing.Spring.height
import sun.swing.SwingUtilities2.drawRect
import com.sun.java.swing.plaf.windows.WindowsGraphicsUtils.paintText
import oracle.jvm.hotspot.jfr.JFR
import java.awt.Color
import java.awt.Color.*
import java.awt.Dimension
import java.awt.Graphics
import java.awt.Rectangle
import javax.swing.JFrame
import javax.swing.JPanel


private fun createCustom(name: String): JButton {
    return object : JButton(name) {

        override fun updateUI() {
            super.updateUI()
            setUI(CustomButtonUI())
        }
    }
}


fun main(args: Array<String>) {
    GameFrame()
}


class CustomButtonUI: BasicButtonUI()  {
    private val BACKGROUND_COLOR = Color(173, 193, 226)
    private val SELECT_COLOR = Color(102, 132, 186)

    override protected fun paintText(g: Graphics, b: AbstractButton, r: Rectangle, t: String) {
        super.paintText(g, b, r, t)
        g.setColor(SELECT_COLOR)
        g.drawRect(r.x, r.y, r.width, r.height)
    }

    override protected fun paintFocus(g: Graphics, b: AbstractButton,
                             viewRect: Rectangle, textRect: Rectangle, iconRect: Rectangle) {
        super.paintFocus(g, b, viewRect, textRect, iconRect)
        g.setColor(Color.blue.darker())
        g.drawRect(viewRect.x, viewRect.y, viewRect.width, viewRect.height)
    }

    override protected fun paintButtonPressed(g: Graphics, b: AbstractButton) {
        if (b.isContentAreaFilled) {
            g.setColor(SELECT_COLOR)
            g.fillRect(0, 0, b.width, b.height)
        }
    }

    override fun installDefaults(b: AbstractButton) {
        super.installDefaults(b)
        b.font = b.font.deriveFont(11f)
        b.background = BACKGROUND_COLOR
    }


}



class GameFrame: JFrame() {
    val newWidth = 600
    val newHeight = 400
    val display: Display

    init {
        preferredSize = Dimension(newHeight,newWidth)
        minimumSize = Dimension(newWidth,newHeight)
        maximumSize = Dimension(newWidth,newHeight)
        isResizable = true
        defaultCloseOperation = JFrame.EXIT_ON_CLOSE
        setLocationRelativeTo(null)
        display = Display()
        add(display)
        isVisible = true

    }

}




class Display: JPanel() {
    var size = 100


    init {

        layout = null
        repaint()
        isVisible = true

    }

    override fun paintComponent(g: Graphics) {
        super.paintComponent(g)
        background = BLACK
        g.color = WHITE
        g.fillRect(0,0,size,size)
    }

}