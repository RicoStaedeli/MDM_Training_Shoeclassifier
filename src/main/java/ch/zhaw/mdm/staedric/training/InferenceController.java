package ch.zhaw.mdm.staedric.training;


    
import java.io.InputStream;
import java.awt.*;
import java.awt.image.*;
import javax.imageio.*;
import java.io.*;

import org.apache.commons.compress.utils.IOUtils;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import java.io.IOException;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

import ai.djl.ModelException;
import ai.djl.modality.cv.Image;
import ai.djl.translate.TranslateException;

@RestController
public class InferenceController {
    Inference inference = new Inference();

      @PostMapping(path = "/analyze")
    public String predict(@RequestParam("image") MultipartFile image) throws Exception {
        inference.predict(image.getBytes());
        return "Okay";
    }
}