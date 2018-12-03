package maximum.industries

import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import java.io.FileInputStream

fun main(args: Array<String>) {
    DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT);

    if (args.contains("-h")) {
        return trainUsage()
    }
    val gameName = args[0]
    val gameSpec = loadSpec(gameName)

    val modelName = getArg(args, "model") ?: "prod_model.${gameName}"
    val model = loadModel(modelName)?.first
    if (model == null) return

    val dataPattern = getArg(args, "data") ?: "data.$gameName"

    var ema_train = 0.0
    fun ema(ema: Double, next: Double, w: Double) =
            if (ema == 0.0) next else w * ema + (1 - w) * next

    val dataReader = object: FileInstanceReader(1.0, 1.0, 1000,
                                                dataPattern, "done") {
        override fun nextStream(): FileInputStream {
            if (ema_train > 0) println(ema_train)
            ema_train = 0.0
            return super.nextStream()
        }
    }

    var valuemult = getArg(args, "valuemult")?.toFloat() ?: 1f
    var maxentropytopfrac = getArg(args, "metf")?.toDouble() ?: 0.0

    while (true) {
        val train_batch = getBatch(gameSpec, dataReader, 500,
                                   true, true, false,
                                   valuemult, maxentropytopfrac)

        val train_score = model.score(train_batch)
        ema_train = ema(ema_train, train_score, 0.9)
    }
}
