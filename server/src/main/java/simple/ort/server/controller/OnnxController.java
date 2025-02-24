package simple.ort.server.controller;

import simple.ort.server.request.AddOnnxRequest;
import simple.ort.server.request.InferImageRequest;
import simple.ort.server.resp.GetOnnxMetaResponse;
import simple.ort.server.service.OnnxService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.Set;

@RestController
public class OnnxController {
    @Autowired
    private OnnxService service;

    @PostMapping("/onnx")
    public Set<String> addOnnx(@RequestBody AddOnnxRequest request) {
        service.add(request.getModelFile(), request.getName());
        return service.listOnnxNames();
    }

    @GetMapping("/onnx/meta")
    public GetOnnxMetaResponse getOnnxMeta(@RequestParam String name) {
        return new GetOnnxMetaResponse(service.inputInfo(name),
                service.outputInfo(name), service.meta(name));
    }

    @PostMapping("/onnx/infer")
    public String inferImage(@RequestBody InferImageRequest request) {
        return service.inferImageB64(request.getImageB64(), request.getName(), request.getInputName());
    }

    @DeleteMapping("/onnx")
    public Set<String> inferImage(@RequestParam String name) {
        service.delete(name);
        return service.listOnnxNames();
    }
}
