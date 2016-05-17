#Opinion miner deluxe ++#

##Introduction##

The opinion miner deluxe "plus plus" is an improved version of the opinion miner based on machine learning that can be trained using a list of
KAF/NAF files. It is important to notice that the opinion miner module will not call
to any external module to obtain features. It will read all the features from the input KAF/NAF file,
so you have to make sure that your input file contains all the required information in advance (tokens,
terms, polarities, constituents, entitiess, dependencies...). This is an example that the opinion miner
would extract from the sentence `I said that the hotel is nice, but the staff is the best!!`:
```xml
    <opinion oid="o2">
      <opinion_holder>
        <!-- I -->
        <span>
          <target id="t1"/>
        </span>
      </opinion_holder>
      <opinion_target>
        <!-- staff -->
        <span>
          <target id="t10"/>
        </span>
      </opinion_target>
      <opinion_expression polarity="DSE" strength="1">
        <!-- the best !! -->
        <span>
          <target id="t12"/>
          <target id="t13"/>
          <target id="t14"/>
        </span>
      </opinion_expression>
    </opinion>
```

The task is general divided into 2 steps
* Detection of opinion entities (holder, target and expression): using
Conditional Random Fields
* Opinion entity linking (expression<-target and expression-<holder): using a simple heuristic (in the near future it will implemented as a
binary Support Vector Machines)

##Quick Installation##

To install this software, just follow these steps:
* git clone https://github.com/rubenIzquierdo/opinion_miner_deluxePP
* cd opinion_miner_deluxePP
* . install_me.sh

The `install_me.sh` will download and compile the required dependencies, and it will also download the trained models. You will be asked
for a password during the download process. To obtain the password please mail me (you will find my contact details at the end of this documentation)
To check if the installation was correct you can run this command:
```
cat example_en.naf | tag_file.py -d hotel
```
You should get a NAF file with opinions in the output. The script will assume to find the models in a predefined path, you can also specify the path to the folder where you have the models with the option `-f`:
```
cat example_en.naf | tag_file.py -f path/to/my_model/
```

##Usage of the opinion tagger##

The main script for tagging opinions is `tag_file.py`. It takes a KAF/NAF file as input stream and it has some parameters. You can see the parameters
by running:
```
tag_file.py -h
usage: tag_file.py [-h] [-v] (-d DOMAIN | -f PATH_TO_FOLDER) [-log]

Detects opinions in KAF/NAF files

optional arguments:
  -h, --help         show this help message and exit
  -v, --version      show program's version number and exit
  -d DOMAIN          Domain for the model (hotel,news)
  -f PATH_TO_FOLDER  Path to a folder containing the model
  -log               Show log information
  -polarity          Run the polarity (positive/negative) classifier too

Example of use: cat example.naf | tag_file.py -d hotel -polarity
```


##Description of the internal process##




In next subsections, a brief explanation of the 2 steps is given.

###Opinion Entity detection###

The first step when extracting opinions from text is to determine which portions of text represent the different opinion entities:

- Opinion expressions: very nice, really ugly ...
- Opinion targets: the hotel, the rooms, the staff ...
- Opinion holders: I, our family, the manager ...

In order to do this, three different Conditional Random Fields (CRF) classifiers have been trained using by default this set of features: tokens,
lemmas, part-of-speech tags, constituent labels and polarity of words and entities. These classifiers detect portions of text representeing differnet opinion
entities.


###Opinion Entity linking###

This step takes as input the opinion entities detected in the previous step, and links them to create the final opinions <expression/target/holder>.
In this case we have trained two binary Support Vector Machines (SVM), one that indicates the degree of association between a given target and a given expression,
and another one that gives the degree of linkage between a holder and an opinion expression. So given a list of expressions, a list of targets and holders detected
by the CRF classifiers, the SVM models try to select the best candidate from the target list for each expressions, and the best holder from the holder list, to create
the final opinion triple.

Considering a certain opinion expression and a target, these are the features by default used to represent this data for the SVM engine:

1) Textual features: tokens and lemmas of the expression and the target
2) Distance features: features representing the relative distance of both elements in the text (normalized to a discrete list of possible values: far/medium/close for instance),
  and if both elements are in the same sentence or not
3) Dependency features: to indicate the dependency relations between the two elements in the text (dependency path, and dependencies relations with the root of the sentence)


##Training##
To be completed...


##Contact##
* Ruben Izquierdo
* Vrije University of Amsterdam
* ruben.izquierdobevia@vu.nl  rubensanvi@gmail.com
* http://rubenizquierdobevia.com/
