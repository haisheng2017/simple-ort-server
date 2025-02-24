package simple.ort.server.request;

import lombok.Data;

@Data
public class AddOnnxRequest {
    private String modelFile;
    private String name = "";
}
