/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package mlspamdetection;

import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import weka.classifiers.trees.J48;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import java.util.Random;


   
    
/**
 *
 * @author SreeRama-PC
 */
public class Mlspamdetection {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws FileNotFoundException, IOException, Exception 
    {
     BufferedReader dataset = new BufferedReader(new FileReader("D:\\spambase.arff"));
     Instances data = new Instances(dataset);
     dataset.close();
     data.setClassIndex(data.numAttributes() - 1);
     data.stratify(10);
    long startT = System.currentTimeMillis(); 
    long T = System.currentTimeMillis() / 1000;
    NaiveBayes bayes = new NaiveBayes();     // naivebayes Object
    J48 tree = new J48();                    // J48 Object
    DecisionTable dt = new DecisionTable();  // Decision tabel Object 
    
    // Training the dataset with each selected classifiers
    /*
      bayes.buildClassifier(train);
      eval.evaluateModel(bayes, test);
      System.out.println(eval.toSummaryString("\n Training Results: NaiveBayes \n======\n", true));
      
      tree.buildClassifier(train);
      eval.evaluateModel(tree, test);
      System.out.println(eval.toSummaryString("\n Training Results: J48 \n======\n", true));
    
      dt.buildClassifier(train);
      eval.evaluateModel(dt, test);
      System.out.println(eval.toSummaryString("\n Training Results: DecisionTree \n======\n", true));
     */
    
     //ten-fold Cross-validation test for naive bayes and displaying results of accuracy and f-measure
    
     double acc_1 =0;
     double meanacc_1 = 0;
     double meanfmeas_1= 0;
     double fmeas_1=0;
     double mean_1=0;
     double stddev_1=0;
     startT = System.currentTimeMillis()/1000;    
     int foldsize_1 = data.numInstances() / 10;
     int foldstart_1 = 0;
     int foldend_1= foldsize_1-1; 
     for (int i=0; i<10; i++)
     {   System.out.println("NaiveB Iteration:"+i);
           Instances train = new Instances(data);                   
           Instances test = new Instances(data, foldstart_1,(foldend_1-foldstart_1));
           NaiveBayes nbayes = new NaiveBayes();
           for (int j=0;j< (foldend_1 - foldstart_1);j++)
           {
                train.delete(foldstart_1);
           }
           nbayes.buildClassifier(test);
           Evaluation evalu = new Evaluation(test);
           evalu.evaluateModel(nbayes, test);
              
           acc_1 = (evalu.numTruePositives(1)+ evalu.numTrueNegatives(1)) / evalu.numInstances();
           fmeas_1 =evalu.fMeasure(1);
           
           System.out.println("Accuracy:" +acc_1);
           System.out.println("Fmeasure:" +fmeas_1);
          
        
           meanacc_1 +=acc_1;
           meanfmeas_1 +=fmeas_1;
           foldstart_1= foldend_1 + 1;
           foldend_1 += foldsize_1;
           if (i==(9))
           { foldend_1 = data.numInstances();
           }
     }
        long end = System.currentTimeMillis()/1000; 
            System.out.println("Training time: "+(end-startT));
            System.out.println("Mean Accuracy: "+meanacc_1/10.0);
            System.out.println("Mean Fmeasure: "+meanfmeas_1/10.0);    
  
//  ten-fold Cross-validation test for J48 and displaying results of accuracy and f-measure
     double acc_2 =0;
     double meanacc_2 = 0;
     double meanfmeas_2= 0;
     double fmeas_2=0;
     double mean_2=0;
     double stddev_2=0;
     startT = System.currentTimeMillis()/1000;
     int foldsize_2 = data.numInstances() / 10;
     int foldstart_2 = 0;
     int foldend_2= foldsize_2-1; 
     for (int i=0; i<10; i++)
     {   System.out.println("J48 Iteration:"+i);
           Instances train = new Instances(data);                   
           Instances test = new Instances(data, foldstart_2,(foldend_2-foldstart_2));
           J48 tree_2 = new J48();
           for (int j=0;j< (foldend_2 - foldstart_2);j++)
           {
                train.delete(foldstart_2);
           }
           tree_2.buildClassifier(test);
           Evaluation evalu = new Evaluation(test);
           evalu.evaluateModel(tree_2, test);
              
           acc_2 = (evalu.numTruePositives(1)+ evalu.numTrueNegatives(1)) / evalu.numInstances();
           fmeas_2 =evalu.fMeasure(0);
           
           System.out.println("Accuracy:" +acc_2);
           System.out.println("Fmeasure:" +fmeas_2);
        
           meanacc_2+=acc_2;
           meanfmeas_2 +=fmeas_2;
           foldstart_2= foldend_2 + 1;
           foldend_2 += foldsize_2;
           if (i==(9))
           { foldend_2 = data.numInstances();
           }
    }   end = System.currentTimeMillis()/1000; 
            System.out.println("Training time: "+(end-startT));
           System.out.println("Mean Accuracy: "+meanacc_2/10.0);
           System.out.println("Mean Fmeasure: "+meanfmeas_2/10.0);    
     
     //ten-fold Cross-validation test for decision table and displaying results of accuracy and f-measure
     double acc =0;
     double meanacc = 0;
     double meanfmeas= 0;
     double fmeas=0;
     double mean=0;
     double stddev=0;
     startT = System.currentTimeMillis()/1000;
     int foldsize = data.numInstances() / 10;
     int foldstart = 0;
     int foldend= foldsize-1; 
     for (int i=0; i<10; i++)
     {   System.out.println("DecisionTable Iteration:"+i);
           Instances train = new Instances(data);                   
           Instances test = new Instances(data, foldstart,(foldend-foldstart));
           DecisionTable dt_1 = new DecisionTable();  
           for (int j=0;j< (foldend - foldstart);j++)
           {
                train.delete(foldstart);
           }
           dt_1.buildClassifier(test);
           Evaluation evalu = new Evaluation(test);
           evalu.evaluateModel(dt_1, test);
             // accuracy = (no.of truepositives = no.of falsepositives)/ total no.of instances 
           acc = (evalu.numTruePositives(1)+ evalu.numTrueNegatives(1)) / evalu.numInstances(); 
           fmeas =evalu.fMeasure(1);
           
           System.out.println("Accuracy:" +acc);
           System.out.println("Fmeasure:" +fmeas);
        
           meanacc +=acc;
           meanfmeas +=fmeas;
           foldstart= foldend + 1;
           foldend += foldsize;
           if (i==(9))
           { foldend = data.numInstances();
           }
     }end = System.currentTimeMillis()/1000; 
            System.out.println("Training time: "+(end-startT));
           System.out.println("Mean Accuracy: "+meanacc/10.0);
           System.out.println("Mean Fmeasure: "+meanfmeas/10.0);    

}
}   

