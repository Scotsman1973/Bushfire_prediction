# -*- coding: utf-8 -*-
#this script relied on two previous posts and their answers.  Both the questions and the answers helped me
#think in different ways about problems I was having with the code.  I also wrote a question on GIS stackExchange which was answered
#I've included the URLs of all three questions in the comments
"""
***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""
import numpy as np
import pandas as pd
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing,
                       QgsFeatureSink,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink,
                       QgsProcessing, QgsFeatureRequest,
                       QgsFeatureSink, QgsProcessingParameterString,
                       QgsProcessingException,
                       QgsProcessingAlgorithm, QgsProcessingParameterVectorDestination,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterRasterDestination,
                       QgsRasterLayer, QgsProcessingContext,
                       QgsFields, QgsField, QgsFeature,
                       QgsVectorLayer, QgsProject, QgsProperty)
from qgis import processing


class ExampleProcessingAlgorithm(QgsProcessingAlgorithm):
    RASTER_INPUT = 'RASTER_INPUT'
    CENTROID_INPUT = 'CENTROID_INPUT'
    VECTER_OUTPUT = 'VECTER_OUTPUT'
    BUFFER_OUTPUT = 'BUFFER OUTPUT'
    Z_STAT_OUTPUT = 'ZONAL STATS OUTPUT'
    Z_HIST_OUTPUT = 'ZONAL HIST OUTPUT'
    VEG_CALC_OUTPUT = 'VEG COVER OUTPUT'
    JOIN_VEG_OUTPUT = 'VEG% AROUND CENTROIDS'
    DROP_OUTPUT = 'DROPPED GEOMS OUTPUT'
    DIST_OUTPUT = 'DISTANCE MATRIX OUTPUT'
    JOIN_DIST_OUTPUT = 'JOIN DISTANCE TO VEGETATION'
    DROP_COMB_OUTPUT = 'DROP GEOM DISTANCE AND VEG COVER'
    JOIN_VEG_DIST_OUTPUT = 'JOIN VEG DIST 2 CENTROIDS'
    RASTER_OUTPUT = 'RASTER_OUTPUT'
    DANGEROUS_PERCENT_VEG = 'DANGEROUS_PERCENT_VEG'
    DIST_TO_NEIGHBOUR = 'DIST_TO_NEIGHBOUR'
    VEG_BUFFER_DIST = 'VEG_BUFFER_DIST'
    WIND_AZIMUTH = 'WIND_AZIMUTH'
    WIND_BAND_WIDTH = 'WIND_BAND_WIDTH'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return ExampleProcessingAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        """
        return 'myscript'

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr('My Script')

    def group(self):
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return self.tr('Example scripts')

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'examplescripts'

    def shortHelpString(self):
        """
        Returns a localised short helper string for the algorithm. This string
        should provide a basic description about what the algorithm does and the
        parameters and outputs associated with it..
        """
        return self.tr("Example algorithm short description")

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """

        # We add the input vector features source. It can have any kind of
        # geometry.
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.CENTROID_INPUT,
                self.tr('Input layer'),
                [QgsProcessing.TypeVectorAnyGeometry]
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.RASTER_INPUT, 'RASTER_INPUT', defaultValue=None
                
            )
        )
        
        self.addParameter(QgsProcessingParameterNumber(
                self.VEG_BUFFER_DIST, self.tr('Distance around each property (metres) assess vegetation cover'), defaultValue = 30 ) )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DANGEROUS_PERCENT_VEG,
                self.tr('Dangerous percentage of vegetation cover'), defaultValue = 30 ) )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DIST_TO_NEIGHBOUR,
                self.tr('Safe distance between buildings (meters)'), defaultValue = 30) )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.WIND_AZIMUTH,
                self.tr('Azimuth of prevailing wind (degrees), N = 0, E = 90, S = 180, W = 270'), defaultValue = 90) )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.WIND_BAND_WIDTH,
                self.tr('Band width of wind direction (degrees)'), defaultValue = 30) )
        # We add a feature sink in which to store our processed features (this
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.VECTER_OUTPUT,
                self.tr('Output layer')
            )
        )
        #rasters are output this way
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                  self.RASTER_OUTPUT, 'RASTER_OUTPUT'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.BUFFER_OUTPUT,
                self.tr('Buffer layer')
            )
        )
        
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.Z_STAT_OUTPUT,
                self.tr('Stats layer')
            )
        )
        
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.Z_HIST_OUTPUT,
                self.tr('Zonal histogram layer')
            )
        )
        
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.VEG_CALC_OUTPUT,
                self.tr('Veg Cover layer')
            )
        )
        
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.DROP_OUTPUT,
                self.tr('Dropped geoms layer')
            )
        )
        
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.JOIN_VEG_OUTPUT,
                self.tr('Vegetation and centroids layer')
            )
        )
        
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.JOIN_DIST_OUTPUT,
                self.tr('Join distance to Vegetation and centroids')
            )
        )
        
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.DIST_OUTPUT,
                self.tr('Distance Matrix output')
            )
        )
        
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.DROP_COMB_OUTPUT,
                self.tr('Dropped geoms layer')
            )
        )
        
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.JOIN_VEG_DIST_OUTPUT,
                self.tr('Dropped geoms layer')
            )
        )
        
    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """
        raster_input = self.parameterAsRasterLayer(parameters, self.RASTER_INPUT, context)
        #################################################################################
        centroids = self.parameterAsVectorLayer(
            parameters,
            self.CENTROID_INPUT,
            context
        )

        # If source was not found, throw an exception to indicate
        if centroids is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.CENTROID_INPUT))
        ###############################################
        buffer_distance = self.parameterAsInt(parameters,
                                            self.VEG_BUFFER_DIST,
                                            context)
                                            
        safe_veg_pcent = self.parameterAsInt(parameters,
                                            self.DANGEROUS_PERCENT_VEG,
                                            context)
                                            
        safe_distance = self.parameterAsInt(parameters,
                                            self.DIST_TO_NEIGHBOUR,
                                            context)
        
        central_azimuth = self.parameterAsInt(parameters,
                                            self.WIND_AZIMUTH,
                                            context)
                                            
        azimuth_band_width = self.parameterAsInt(parameters,
                                            self.WIND_BAND_WIDTH,
                                            context)
        
        ################################################
        #I spent 2 weeks attempting to work with raster data within a processing script, without luck, but then I found this answer
        #https://gis.stackexchange.com/questions/398422/passing-parameters-to-qgisrastercalculator-alg-in-custom-processing-script?rq=1
        #I used it exactly, because it is the only way I have found that works
        #####################################################################################
        output = self.parameterAsFileOutput(parameters, self.RASTER_OUTPUT, context)
        params = {
            'CELLSIZE': 0,
            'CRS': None,
            'EXPRESSION': '\"{}@1\"'.format(raster_input.name()),
            'EXTENT': None,
            'LAYERS': [raster_input],
            'OUTPUT':  parameters['RASTER_OUTPUT'],
        }
        result = processing.run('qgis:rastercalculator', params, context=context)
        ########################
        #there is some pretty funky syntax in this script, from here on it is inspired by this post on StackExchange
        #https://gis.stackexchange.com/questions/387195/buffer-algorithm-doesnt-work-in-a-new-script-from-template
        #the problem it solved was that by buffering centroids the result had no geometry and so all operations going forward weren't possible
        #I experimented with alternative methods for using vectors and having vectors and rasters interact with each other
        ###################################################################################################
        outputFile = self.parameterAsOutputLayer(parameters, self.BUFFER_OUTPUT, context)
        buffered_layer = processing.run("native:buffer", {
            'INPUT': centroids,
            'DISTANCE': 30,
            'SEGMENTS': 5,
            'END_CAP_STYLE': 0,
            'JOIN_STYLE': 0,
            'MITER_LIMIT': 2,
            'DISSOLVE': False,
            'OUTPUT': outputFile
        }, is_child_algorithm=True, context=context, feedback=feedback)['OUTPUT']
        
        output_Z_stat = self.parameterAsOutputLayer(parameters, self.Z_STAT_OUTPUT, context)
        zonal_stat = processing.run("native:zonalstatisticsfb", {
        'INPUT': buffered_layer,
        'INPUT_RASTER': result['OUTPUT'],
        'RASTER_BAND':1,
        'COLUMN_PREFIX':'_',
        'STATISTICS':[0,1,2],
        'OUTPUT': output_Z_stat
        },
        context=context, feedback=feedback, is_child_algorithm=True)['OUTPUT']
        
        output_Z_hist = self.parameterAsOutputLayer(parameters, self.Z_HIST_OUTPUT, context)
        zonal_hist = processing.run("native:zonalhistogram",{
        'INPUT_RASTER': result['OUTPUT'],
        'RASTER_BAND': 1, # where the land cover info is stored
        'INPUT_VECTOR': zonal_stat, # chains buffering through zonal stats, saves the step of joining the two attribute tables
        'COLUMN_PREFIX': 'HISTO_', # creates new columns prefixed with 'HISTO_'. doesn't mean anything
        'OUTPUT': output_Z_hist # stores the results in a .gpkg scratch layer
        },
        context=context, feedback=feedback, is_child_algorithm=True)['OUTPUT']
        
        output_calc_1 = self.parameterAsOutputLayer(parameters, self.VEG_CALC_OUTPUT, context)
        veg_calc_output = processing.run("qgis:fieldcalculator", {
            'INPUT': zonal_hist,
            'FIELD_NAME': 'VEG%',
            'FIELD_TYPE': 2, #0 is float type, 2 is string
            'FIELD_LENGTH': 25,
            'FIELD_PRECISION': 2,
            'FORMULA': '  (  "HISTO_5"  /   "_count"  )   * 100',  
            'OUTPUT': output_calc_1
        },
        context=context, feedback=feedback, is_child_algorithm=True)['OUTPUT']
        
        drop_geoms_output = self.parameterAsOutputLayer(parameters, self.DROP_OUTPUT, context)
        dropped_geoms = processing.run("native:dropgeometries", {
        'INPUT': veg_calc_output,
        'OUTPUT': drop_geoms_output
        },
        context=context, feedback=feedback, is_child_algorithm=True)['OUTPUT']
        
        join_veg_output = self.parameterAsOutputLayer(parameters, self.JOIN_VEG_OUTPUT, context)
        join_veg = processing.run("native:joinattributestable", {
        'INPUT': centroids,
        'FIELD':'OBJECTID',
        'INPUT_2': dropped_geoms,
        'FIELD_2':'OBJECTID',
        'FIELDS_TO_COPY':['VEG%'],
        'METHOD':1,
        'DISCARD_NONMATCHING':False,
        'PREFIX':'',
        'OUTPUT': join_veg_output
        },
        context=context, feedback=feedback, is_child_algorithm=True)['OUTPUT']
        
        dist_output = self.parameterAsOutputLayer(parameters, self.DIST_OUTPUT, context)
        dist_result = processing.run("qgis:distancematrix",{
        'INPUT': join_veg,
        'INPUT_FIELD': 'OBJECTID',
        'TARGET': join_veg,
        'TARGET_FIELD': 'OBJECTID',
        'MATRIX_TYPE': 0,
        'NEAREST_POINTS': 1,
        'OUTPUT': dist_output
        },
        context=context, feedback=feedback, is_child_algorithm=True)['OUTPUT']
        
        join_dist_output = self.parameterAsOutputLayer(parameters, self.JOIN_DIST_OUTPUT, context)
        join_dist = processing.run("native:joinattributestable", {
        'INPUT': join_veg,
        'FIELD':'OBJECTID',
        'INPUT_2': dist_result,
        'FIELD_2':'InputID',
        'FIELDS_TO_COPY':['Distance'],
        'METHOD':1,
        'DISCARD_NONMATCHING':False,
        'PREFIX':'',
        'OUTPUT': join_dist_output
        },
        context=context, feedback=feedback, is_child_algorithm=True)['OUTPUT']
        
        drop_geoms_comb = self.parameterAsOutputLayer(parameters, self.DROP_COMB_OUTPUT, context)
        dropped_geoms_comb = processing.run("native:dropgeometries", {
        'INPUT': join_dist,
        'OUTPUT': drop_geoms_comb
        },
        context=context, feedback=feedback, is_child_algorithm=True)['OUTPUT']
        
        #join_dist_veg_output = self.parameterAsOutputLayer(parameters, self.JOIN_VEG_DIST_OUTPUT, context)
        join_dist_veg = processing.run("native:joinattributestable", {
        'INPUT': centroids,
        'FIELD':'OBJECTID',
        'INPUT_2': dropped_geoms_comb,
        'FIELD_2':'OBJECTID',
        'FIELDS_TO_COPY':['VEG%' , 'Distance'],
        'METHOD':1,
        'DISCARD_NONMATCHING':False,
        'PREFIX':'',
        'OUTPUT': 'memory:'
        },
        context=context, feedback=feedback, is_child_algorithm=True)['OUTPUT']
        centroids = context.getMapLayer(join_dist_veg)
        
        #######################################################################
        #the next section of code computes the azimuth between all structures, and the structure closest to them
        #this section is drawn from an answer to a question I asked.  This is the URL:
        #
        
        #######################################################################
        #the next section of code computes the azimuth between all structures, and the structure closest to them
        #this section is drawn from an answer to a question I asked.  This is the URL:
        #
        layer = centroids
        feats = [ feat for feat in layer.getFeatures() ]
        n = len(feats)
        distances = [ [] for i in range(n) ]
        indices = [ [] for i in range(n) ]
        ##determining distances and indices for each pair of points
        ##itertools python module (for avoiding repeated distances values) produces wrong results
        for i in range(n):
            for j in range(n):
                if i != j:#makes sure i and j are different
                    distances[i].append(feats[i].geometry().distance(feats[j].geometry()))
                    indices[i].append([i,j])
        
        min_distance = []

        #determining min distance for each pair of points
        for i, item in enumerate(distances):
            min_distance.append((np.min(item)))
            
        index_dist = []

        #determining index for min distance in distances list
        for i, item in enumerate(distances):
            for j, element in enumerate(item):
                if min_distance[i] == element:
                    index_dist.append(j)

        index_closest_pairs = []
        correct_azimuths = []
        #determining indices for min distance (closest pairs) in indices list
        for i, index in enumerate(indices):
            index_closest_pairs.append(index[index_dist[i]])
        
        #azimuth_output = join_dist #this line is needed so that the field calculator can enter the update loop and go through each feature in turn
        row = 0
        #determining azimuth for closest_pairs, the function in QGIS doesn't work properly
        for item in index_closest_pairs:
            correct_azimuth = (feats[item[0]].geometry().asPoint().azimuth(feats[item[1]].geometry().asPoint()))
            
            if (correct_azimuth < 0):
                final_azimuth =  (correct_azimuth + 360)
                
            else:
                final_azimuth = correct_azimuth
            
            correct_azimuths.append(final_azimuth)
            row+=1
        ###########################
        #these next 20 lines take 2 lists, zip them together, make a dataframe, export as csv, then import the csv and join one field
        #this is how the site prediction model and machine learning will be made to work
        #this is definately the steps that can best update 
        objectid_df = []
        for feature in layer.getFeatures():
            new = feature["OBJECTID"]#need to have unique identifier to join attributes
            objectid_df.append(new)
            
        azimuth_df = pd.DataFrame( list ( zip ( objectid_df , correct_azimuths) ) , columns=[ 'OBJECTID' , 'azimuth'])
        azimuth_df.to_csv( 'C:\Mythings\Data\canbera_bushfire\output_azimuth.csv', index=False )
        ################################################
        #use join attribute table to load a layer midway through a processing script
        #need to concatonate the string
        join_dist_veg_az = processing.run("native:joinattributestable", {
        'INPUT': centroids,
        'FIELD':'OBJECTID',
        'INPUT_2': 'delimitedtext://file:///'+'C:/Mythings/Data/canbera_bushfire/output_azimuth.csv'+'?type=csv&maxFields=10000&detectTypes=yes&geomType=none&subsetIndex=no&watchFile=no',
        'FIELD_2':'OBJECTID',
        'FIELDS_TO_COPY':['azimuth'],
        'METHOD':1,
        'DISCARD_NONMATCHING':False,
        'PREFIX':'',
        'OUTPUT': 'memory:'
        },
        context=context, feedback=feedback, is_child_algorithm=True)['OUTPUT']
        centroids1 = context.getMapLayer(join_dist_veg_az)
        
        vuln_df = []
        veg_pc_df = []
        dist_df = []
        for feature in centroids1.getFeatures():
            veg = feature["VEG%"]
            veg_pc_df.append(veg)
            dist = feature["Distance"]
            dist_df.append(dist)
            
        for i, veg_pcent in enumerate(veg_pc_df):
        
            if (float(dist_df[i]) <=  safe_distance and float(veg_pc_df[i]) > safe_veg_pcent ):
                v_d_vuln = 'Medium'
            
            else:
                v_d_vuln = 'Low'
                
            vuln_df.append(v_d_vuln)
        
        vuln_id_df = pd.DataFrame( list ( zip ( objectid_df , vuln_df) ) , columns=[ 'OBJECTID' , 'vuln'])
        vuln_id_df.to_csv( 'C:\Mythings\Data\canbera_bushfire\output_vulnerability.csv', index=False )
        ################################################
        #use join attribute table to load a layer midway through a processing script
        #need to concatonate the string
        join_vuln = processing.run("native:joinattributestable", {
        'INPUT': centroids1,
        'FIELD':'OBJECTID',
        'INPUT_2': 'delimitedtext://file:///'+'C:/Mythings/Data/canbera_bushfire/output_vulnerability.csv'+'?type=csv&maxFields=10000&detectTypes=yes&geomType=none&subsetIndex=no&watchFile=no',
        'FIELD_2':'OBJECTID',
        'FIELDS_TO_COPY':['vuln'],
        'METHOD':1,
        'DISCARD_NONMATCHING':False,
        'PREFIX':'',
        'OUTPUT': 'memory:'
        },
        context=context, feedback=feedback, is_child_algorithm=True)['OUTPUT']
        centroids1 = context.getMapLayer(join_vuln)
        ############################
        wind_dir = []
        ###############
        azimuth_band_side = azimuth_band_width / 2
        #
        complicated_from = 360 - azimuth_band_side
        complicated_to = 0 + azimuth_band_side
        azimuth_from = central_azimuth - azimuth_band_side
        azimuth_to = central_azimuth + azimuth_band_side
        #
        #these two if statements correct the azimuth after additions/deductions that might
        #otherwise mean that the result is greater than 360 or less than 0
        if (azimuth_to > 360):
            azimuth_to = azimuth_to - 360
            
        if (azimuth_from < 0):
            azimuth_from = azimuth_from + 360
        ############################
        for i, azims in enumerate(correct_azimuths):
        
            if ((0 < azimuth_to < complicated_to) or (complicated_from < azimuth_from < 360)):
                
                if (( azimuth_from < float(correct_azimuths[i]) < 360) or (0 < float(correct_azimuths[i]) < azimuth_to)):
                    wind_direction = 'Dangerous'
                
                else:
                    wind_direction = 'Not dangerous'
                    
            else:
                    
                if (float(correct_azimuths[i]) > azimuth_from) and (float(correct_azimuths[i]) < azimuth_to):
                    wind_direction = 'Dangerous'
                
                else:
                    wind_direction = 'Not dangerous'
            
                wind_dir.append(wind_direction)
        
        wind_dir_id_df = pd.DataFrame( list ( zip ( objectid_df , wind_dir) ) , columns=[ 'OBJECTID' , 'wind_dir'])
        wind_dir_id_df.to_csv( 'C:\Mythings\Data\canbera_bushfire\output_wind_dir.csv', index=False )
        ################################################
        #use join attribute table to load a layer midway through a processing script
        #need to concatonate the string
        join_wind = processing.run("native:joinattributestable", {
        'INPUT': centroids1,
        'FIELD':'OBJECTID',
        'INPUT_2': 'delimitedtext://file:///'+'C:\Mythings\Data\canbera_bushfire\output_wind_dir.csv'+'?type=csv&maxFields=10000&detectTypes=yes&geomType=none&subsetIndex=no&watchFile=no',
        'FIELD_2':'OBJECTID',
        'FIELDS_TO_COPY':['wind_dir'],
        'METHOD':1,
        'DISCARD_NONMATCHING':False,
        'PREFIX':'',
        'OUTPUT': 'memory:'
        },
        context=context, feedback=feedback, is_child_algorithm=True)['OUTPUT']
        centroids1 = context.getMapLayer(join_wind)
        ###################################
        dist_df = []
        for feature in centroids1.getFeatures():
            dist = feature["wind_dir"]
            dist_df.append(dist)
        ######################
        updated_vuln = []
        ############################
        for i, azims in enumerate(objectid_df):
        
            if (dist_df[i] == 'Dangerous' and float(veg_pc_df[i]) > safe_veg_pcent):
                updated_vulnrability = 'High vulnerability'
            elif (dist_df[i] == 'Dangerous' and float(veg_pc_df[i]) < safe_veg_pcent):
                updated_vulnrability = 'Medium vulnerability'
            else:
                updated_vulnrability = 'Vulerability unchanged'
            
            updated_vuln.append(updated_vulnrability)
        
        updated_vuln_df = pd.DataFrame( list ( zip ( objectid_df , updated_vuln) ) , columns=[ 'OBJECTID' , 'updated_vuln'])
        updated_vuln_df.to_csv( 'C:\Mythings\Data\canbera_bushfire\output_updated_vuln.csv', index=False )
        ################################################
        #use join attribute table to load a layer midway through a processing script
        #need to concatonate the string
        join_updated_vuln = processing.run("native:joinattributestable", {
        'INPUT': centroids1,
        'FIELD':'OBJECTID',
        'INPUT_2': 'delimitedtext://file:///'+'C:\Mythings\Data\canbera_bushfire\output_updated_vuln.csv'+'?type=csv&maxFields=10000&detectTypes=yes&geomType=none&subsetIndex=no&watchFile=no',
        'FIELD_2':'OBJECTID',
        'FIELDS_TO_COPY':['updated_vuln'],
        'METHOD':1,
        'DISCARD_NONMATCHING':False,
        'PREFIX':'',
        'OUTPUT': 'memory:'
        },
        context=context, feedback=feedback, is_child_algorithm=True)['OUTPUT']
        centroids1 = context.getMapLayer(join_updated_vuln)
        #############################
        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.VECTER_OUTPUT,
            context,
            centroids1.fields(),
            centroids.wkbType(),
            centroids.sourceCrs()
        )

        # Send some information to the user
        feedback.pushInfo('CRS is {}'.format(centroids.sourceCrs().authid()))

        # If sink was not created, throw an exception to indicate that the algorithm
        # encountered a fatal error. The exception text can be any string, but in this
        # case we use the pre-built invalidSinkError method to return a standard
        # helper text for when a sink cannot be evaluated
        if sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters, self.VECTER_OUTPUT))

        ####################
         # Compute the number of steps to display within the progress bar and
        # get features from source
        total = 100.0 / centroids.featureCount() if centroids.featureCount() else 0
        features = centroids1.getFeatures()
        
        for current, feature in enumerate(features):
            # Stop the algorithm if cancel button has been clicked
            if feedback.isCanceled():
                break
        ##################################################
        ##################################################ADD FEATURES TO SINK
        #####################################################
            # Add a feature in the sink
            sink.addFeature(feature, QgsFeatureSink.FastInsert)

            # Update the progress bar
            feedback.setProgress(int(current * total))

        # Return the results of the algorithm. In this case our only result is
        # the feature sink which contains the processed features, but some
        # algorithms may return multiple feature sinks, calculated numeric
        # statistics, etc. These should all be included in the returned
        # dictionary, with keys matching the feature corresponding parameter
        # or output names.
        
        results = {}
        results[self.RASTER_OUTPUT] = result['OUTPUT']
        results[self.VECTER_OUTPUT] = dest_id
        #results[self.BUFFER_OUTPUT] = buffered_layer
        #results[self.Z_STAT_OUTPUT] = zonal_stat
        results[self.Z_HIST_OUTPUT] = zonal_hist
        results[self.VEG_CALC_OUTPUT] = veg_calc_output
        results[self.DROP_OUTPUT] = dropped_geoms
        results[self.JOIN_VEG_OUTPUT] = join_veg 
        results[self.DIST_OUTPUT] = dist_result
        return results
