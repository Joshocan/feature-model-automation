import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.charset.StandardCharsets;
import de.ovgu.featureide.fm.core.base.IFeatureModel;
import de.ovgu.featureide.fm.core.base.impl.DefaultFeatureModelFactory;
import de.ovgu.featureide.fm.core.base.impl.FMFactoryManager;
import de.ovgu.featureide.fm.core.io.Problem;
import de.ovgu.featureide.fm.core.io.ProblemList;
import de.ovgu.featureide.fm.core.io.xml.XmlFeatureModelFormat;

/** Read-only bridge for FeatureIDE 3.10.0. One isolated JVM per model. */
public class FeatureIDEParse {
    private static String quote(String s) {
        StringBuilder out = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            if (c == '\\' || c == '"') out.append('\\').append(c);
            else if (c < 32) out.append(String.format("\\u%04x", (int)c));
            else out.append(c);
        }
        return out.append('"').toString();
    }
    public static void main(String[] args) throws Exception {
        System.setProperty("javax.xml.accessExternalDTD", "");
        System.setProperty("javax.xml.accessExternalSchema", "");
        String xml = new String(Files.readAllBytes(Paths.get(args[0])), StandardCharsets.UTF_8);
        // External entities and DTD expansion are outside this evaluation's scope.
        if (xml.contains("<!DOCTYPE") || xml.contains("<!ENTITY"))
            throw new IllegalArgumentException("DTD/entity declarations unsupported by batch safety policy");
        FMFactoryManager.getInstance().addExtension(DefaultFeatureModelFactory.getInstance());
        IFeatureModel model = DefaultFeatureModelFactory.getInstance().create();
        ProblemList problems = new XmlFeatureModelFormat().read(model, xml);
        boolean accepted = !problems.containsError() && model.getStructure().getRoot() != null;
        StringBuilder details = new StringBuilder("[");
        for (Problem p : problems) {
            if (details.length() > 1) details.append(',');
            details.append("{\"severity\":").append(quote(p.getSeverity().toString()))
                .append(",\"line\":").append(p.getLine())
                .append(",\"message\":").append(quote(p.getMessage())).append('}');
        }
        details.append(']');
        System.out.println("FEATUREIDE_RESULT " + "{\"parsed_ok\":" + accepted
            + ",\"errors\":" + problems.getErrors().size()
            + ",\"warnings\":" + problems.getWarnings().size()
            + ",\"loaded_features\":" + model.getNumberOfFeatures()
            + ",\"problems\":" + details + "}");
    }
}
