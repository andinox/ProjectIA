package iut.nombre;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.CvType;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.io.File;
import java.net.URL;

public class ProjectIA {

    private static List<String> loadLabels(String labelsPath) {
        List<String> labels = new ArrayList<>();
        try (InputStream inputStream = ProjectIA.class.getResourceAsStream(labelsPath)) {
            if (inputStream != null) {
                BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
                String line;
                while ((line = reader.readLine()) != null) {
                    labels.add(line.trim());
                }
            } else {
                System.err.println("Fichier de labels non trouvé : " + labelsPath);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return labels;
    }


    private static String interpretResults(Mat predictions, List<String> labels) {
        int classId;
        double confidence;

        predictions = predictions.reshape(1, 1); // Redimensionner pour faciliter l'accès aux éléments

        Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(predictions);
        classId = (int) minMaxLocResult.maxLoc.x;
        confidence = minMaxLocResult.maxVal;

        String label = labels.get(classId);
        return "Classe : " + label + " (Confiance : " + confidence + ")";
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        SwingUtilities.invokeLater(() -> {
            createAndShowGUI();
        });
    }

    private static void createAndShowGUI() {
        JFrame frame = new JFrame("Project IA v1.0");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);

        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BorderLayout());

        JLabel imageLabel = new JLabel();
        JLabel nameLabel = new JLabel("Nom de l'image : ");
        JLabel resultLabel = new JLabel("Résultat de la comparaison : ");

        mainPanel.add(imageLabel, BorderLayout.CENTER);
        mainPanel.add(nameLabel, BorderLayout.SOUTH);
        mainPanel.add(resultLabel, BorderLayout.NORTH);

        JButton openButton = new JButton("Ouvrir une image");
        openButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                JFileChooser fileChooser = new JFileChooser();
                FileNameExtensionFilter filter = new FileNameExtensionFilter("Images", "jpg", "jpeg", "png", "gif");
                fileChooser.setFileFilter(filter);

                int result = fileChooser.showOpenDialog(frame);
                if (result == JFileChooser.APPROVE_OPTION) {
                    File selectedFile = fileChooser.getSelectedFile();
                    compareImages(selectedFile, imageLabel, nameLabel, resultLabel);
                }
            }
        });

        frame.add(mainPanel, BorderLayout.CENTER);
        frame.add(openButton, BorderLayout.SOUTH);

        frame.setVisible(true);
    }

    private static void compareImages(File selectedFile, JLabel imageLabel, JLabel nameLabel, JLabel resultLabel) {

        URL modelUrl = ProjectIA.class.getResource("/mobilenet_with-preprocessing.pb");
        String modelPath = new File(modelUrl.getPath()).getAbsolutePath();
        Net net = Dnn.readNetFromTensorflow(modelPath);

        ImageIcon originalIcon = new ImageIcon(selectedFile.getAbsolutePath());
        Image originalImage = originalIcon.getImage();

        int maxWidth = 700;
        int maxHeight = 500;
        Image scaledImage = originalImage.getScaledInstance(maxWidth, maxHeight, Image.SCALE_SMOOTH);

        ImageIcon scaledIcon = new ImageIcon(scaledImage);
        imageLabel.setIcon(scaledIcon);
        nameLabel.setText("Nom de l'image : " + selectedFile.getName());


        Mat inputImage = Imgcodecs.imread(selectedFile.getAbsolutePath());
        Imgproc.resize(inputImage, inputImage, new Size(224, 224));
        inputImage.convertTo(inputImage, CvType.CV_32F);
        Scalar mean = new Scalar(127.5, 127.5, 127.5);
        Core.subtract(inputImage, mean, inputImage);


        Mat blob = Dnn.blobFromImage(inputImage);
        net.setInput(blob);
        Mat predictions = net.forward();


        List<String> labels = loadLabels("/labels.txt");

        String predictionResult = interpretResults(predictions, labels);

        resultLabel.setText("Résultat de la comparaison : " + predictionResult);
    }
}
