package max.dillon;

import com.google.protobuf.TextFormat;
import jdk.nashorn.internal.ir.debug.ClassHistogramElement;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class App {
    public static void main(String[] args) throws IOException {
        String str = new String(Files.readAllBytes(Paths.get("src/main/data/chess.textproto")));
        GameGrammar.GameSpec.Builder builder = GameGrammar.GameSpec.newBuilder();
        TextFormat.getParser().merge(str, builder);
        GameGrammar.GameSpec game = builder.build();

        System.out.println(game.getName());
        System.out.println(game.getBoardSize());
        for (GameGrammar.Piece piece : game.getPieceList()) {
            System.out.println(piece.getName()+" "+piece.getPlacementList());
        }
    }
}

