package com.example.decisiontree;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.util.Enumeration;
import java.util.Iterator;
import java.util.Vector;

/**
 * Implementation with Weka of the example from this
 * <a href="https://dzone.com/articles/machine-learning-with-decision-trees">article</a>
 *
 */
public class App
{
    public static void main( String[] args ) throws Exception
    {
        // load data from CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(ClassLoader.getSystemResourceAsStream("rules.csv"));
        Instances data = loader.getDataSet();

        //initialize the info gain extractor
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        Ranker search = new Ranker();

        AttributeSelection attSelect = new AttributeSelection();
        attSelect.setEvaluator(eval);
        attSelect.setSearch(search);
        attSelect.SelectAttributes(data);

        // let's show the information gain value for each attribute
        System.out.println(attSelect.toResultsString());

        //now we build and show the decision tree
        J48 tree = new J48();
        tree.buildClassifier(data);
        System.out.println(tree.graph());

        Iterator<String> measures =  tree.enumerateMeasures().asIterator();
        while(measures.hasNext()) {
            System.out.println(measures.next());
        }
    }
}