package simple.ort.server.service;

import ai.onnxruntime.*;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Service;
import simple.ort.server.model.OnnxWrapper;

import java.nio.ByteBuffer;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

@Service
public class OnnxService {

    private static final Map<String, OnnxWrapper> ONNX = new LinkedHashMap<>();

    public void add(String modelPath, String name) {
        if (name.isEmpty()) {
            name = modelPath;
        }
        if (ONNX.containsKey(name)) {
            return;
        }
        var env = OrtEnvironment.getEnvironment();
        var options = new OrtSession.SessionOptions();
        // configure a gpu runtime
//        options.addCUDA(0);
        OrtSession session = null;
        try {
            session = env.createSession(modelPath, options);
            ONNX.put(name, new OnnxWrapper(session));
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
    }

    public Set<String> listOnnxNames() {
        return ONNX.keySet();
    }

    public Map<String, NodeInfo> inputInfo(String name) {
        var wrapper = ONNX.get(name);
        try {
            return wrapper.session().getInputInfo();
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
    }

    public Map<String, NodeInfo> outputInfo(String name) {
        var wrapper = ONNX.get(name);
        try {
            return wrapper.session().getOutputInfo();
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
    }

    public OnnxModelMetadata meta(String name) {
        var wrapper = ONNX.get(name);
        try {
            return wrapper.session().getMetadata();
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
    }

    public String inferImageB64(String imageB64, String name, String inputName) {
        byte[] img = Base64.getDecoder().decode(imageB64);
        var wrapper = ONNX.get(name);
        OnnxTensor tensor;
        try {
            var nodeInfo = wrapper.session().getInputInfo().get(inputName);
            TensorInfo info = (TensorInfo) nodeInfo.getInfo();
            tensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(),
                    ByteBuffer.wrap(img),
                    info.getShape(), OnnxJavaType.UINT8);
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
        try (var result = wrapper.session().run(
                Map.of("input", tensor)
        )) {
            Map<String, Object> ret = new LinkedHashMap<>();
            for (var r : result) {
                if (r.getValue().getType() != OnnxValue.OnnxValueType.ONNX_TYPE_TENSOR) {
                    throw new RuntimeException("Unsupported onnx value type: " +
                            r.getKey() + "," + r.getValue().getType());
                }
                var info = (TensorInfo) r.getValue().getInfo();
                // TODO cast value according to different shape
                if (info.type == OnnxJavaType.FLOAT) {
                    if (info.getShape().length == 3) {
                        float[][][] f = (float[][][]) r.getValue().getValue();
                        ret.put(r.getKey(), f);
                    } else {
                        float[][] f = (float[][]) r.getValue().getValue();
                        ret.put(r.getKey(), f);
                    }
                } else {
                    long[][] l = (long[][]) r.getValue().getValue();
                    ret.put(r.getKey(), l);
                }
            }
            return new ObjectMapper().writeValueAsString(ret);
        } catch (OrtException | JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    public void delete(String name) {
        var wrapper = ONNX.remove(name);
        if (wrapper != null) {
            try {
                wrapper.session().close();
            } catch (OrtException e) {
                throw new RuntimeException(e);
            }
        }
    }
}
