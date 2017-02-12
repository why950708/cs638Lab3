import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.Stream;

/**
 * Created by Zirui Tao on 2/12/2017.
 */
public class Lab2 {
    public static void main(String[] args) {
        String FileName = args[1];
        ParseFile(FileName);
    }

    private static Striains[] ParseFile(String fileName) {
        String [] lines =  null;

        try{
            Stream <String> s = Files.lines(Paths.get(fileName));
            // convert into String arrays
            lines = s.toArray(size -> new String[size]);


        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;

    }
}

class Striains {
    
    public Striains() {
        
    }
}
