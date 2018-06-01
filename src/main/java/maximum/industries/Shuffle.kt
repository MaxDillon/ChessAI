package maximum.industries

import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.attribute.BasicFileAttributes
import java.util.ArrayList
import java.util.function.BiPredicate

fun main(args: Array<String>) {
    val inpattern = args[0]
    val outpattern = args[1]

    val extension = getArg(args, "extension") ?: "done"
    var idx = getArg(args, "startidx")?.toInt() ?: 0

    var states = 0
    var error = 0

    val instances = ArrayList<Instance.TrainingInstance>()

    fun SaveInstances() {
        instances.shuffle()
        val outstream = FileOutputStream("$outpattern.${(idx++).toString().padStart(8, '0')}.$extension")
        for (instance in instances) {
            instance.writeDelimitedTo(outstream)
        }
        outstream.close()
        instances.clear()
    }

    val matcher = BiPredicate<Path, BasicFileAttributes> { file, _ ->
        file.toString().matches(Regex(".*" + inpattern))
    }
    for (file in Files.find(Paths.get("."), 1, matcher).sorted().iterator()) {
        println("Processing ${file}")
        val instream = FileInputStream(file.toString())

        while (true) {
            try {
                val newInstance = Instance.TrainingInstance.parseDelimitedFrom(instream) ?: break
                instances.add(newInstance)
                if (instances.size == 10000) {
                    SaveInstances()
                }
                states += 1
            } catch (e: Exception) {
                e.printStackTrace()
                error += 1
            }
        }
    }
    SaveInstances()
    println("Shuffled $states states into $idx files. $error errors.")
}

