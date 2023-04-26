package ch.zhaw.mdm.staedric.training;


import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;



@RestController
public class InferenceController {
    Inference inference = new Inference();

      @PostMapping(path = "/analyze-rico")
    public String predict(@RequestParam("image") MultipartFile image) throws Exception {
        String result = inference.predict(image.getBytes());
        return result;
    }
}