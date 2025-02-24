package simple.ort.server.request;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class InferImageRequest {
    @NotBlank
    private String name;
    @NotBlank
    private String imageB64;

    private String inputName;
}
