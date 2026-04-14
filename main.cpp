//
//  main.cpp
//  gctb
//
//  Created by Jian Zeng on 14/06/2016.
//  Copyright © 2016 Jian Zeng. All rights reserved.
//

#include <cmath>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "gctb.hpp"
#include "quantizer.hpp"
#include "xci.hpp"
#include "vgmaf.hpp"

using namespace std;

#ifdef __linux__
inline int getMemPeakKB() {
    ifstream file("/proc/self/status");
    if (!file.is_open()) return -1;

    string line;
    while (getline(file, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            auto pos = line.find_first_of("0123456789");
            if (pos == string::npos) return -1;

            istringstream iss(line.substr(pos));
            int value = 0;
            iss >> value;
            return value;
        }
    }
    return -1;
}

inline int getVMPeakKB() {
    ifstream file("/proc/self/status");
    if (!file.is_open()) return -1;

    string line;
    while (getline(file, line)) {
        if (line.rfind("VmPeak:", 0) == 0) {
            auto pos = line.find_first_of("0123456789");
            if (pos == string::npos) return -1;

            istringstream iss(line.substr(pos));
            int value = 0;
            iss >> value;
            return value;
        }
    }
    return -1;
}
#endif


int main(int argc, const char * argv[]) {
    
    cout << "*********************************************************\n";
    cout << "* GCTB 2.5.5                                            *\n";
    cout << "* Genome-wide Complex Trait Bayesian analysis           *\n";
    cout << "* For inquiries, contact: Jian Zeng <j.zeng@uq.edu.au>  *\n";
    cout << "* Last updated: 12 Dec, 2025                            *\n";
    cout << "* MIT License                                           *\n";
    cout << "*********************************************************\n";
    
    Gadget::Timer timer;
    timer.setTime();
    cout << "\nAnalysis started: " << timer.getDate();
    
    if (argc < 2){
        cerr << " \nDid you forget to give the input parameters?\n" << endl;
        exit(1);
    }
    
    const string earlyStopToken = "__EARLY_STOP_SBayesR__";
    const bool sbayesrEarlyStop = []() {
        const char *v = std::getenv("GCTB_SBAYESR_EARLY_STOP");
        if (!v) return false;
        string s(v);
        for (auto &c : s) c = static_cast<char>(std::tolower(c));
        return (s == "1" || s == "true" || s == "yes" || s == "on");
    }();
    
    try {
        
        Options opt;
        opt.inputOptions(argc, argv);

        if (opt.analysisType == "QuantizeEigen") {
            eigen_quantize::QuantizationOptions qopt;
            qopt.bits = opt.quantEigenBits;
            qopt.entropy_coding = opt.quantEigenEntropy;
            qopt.q_per_snp_column = opt.quantEigenQPerSnp;
            try {
                const auto summary = eigen_quantize::quantize_directory(opt.quantEigenInputDir, opt.quantEigenOutputDir, qopt);
                cout << "Completed " << summary.num_files << " files with q" << qopt.bits;
                if (qopt.q_per_snp_column) {
                    cout << " (Q per SNP col)";
                }
                if (qopt.entropy_coding) {
                    cout << " (entropy-coded)";
                }
                cout << ".\nTotal bytes: " << summary.total_original_bytes << " -> " << summary.total_quantized_bytes << endl;
            } catch (const std::exception& error) {
                cerr << "Quantize eigen error: " << error.what() << endl;
                return 1;
            }
            timer.getTime();
            cout << "\nAnalysis finished: " << timer.getDate();
            cout << "Computational time: " << timer.format(timer.getElapse()) << endl;
            return 0;
        }
        
        if (opt.seed) Stat::seedEngine(opt.seed);
        else          Stat::seedEngine(011415);  // fix the random seed if not given due to the use of MPI
        
//        cout << "==========" << opt.seed << " " << Stat::ranf() << " " << Stat::snorm() << endl;
        
        Data data;
        data.title = opt.title;
        bool readGenotypes;
        
        GCTB gctb(opt);


        if (opt.analysisType == "Bayes") {
            if (opt.numChains > 1) {
                throw(" Error: multi-chain MCMC is not yet available for individual-level-data analysis.");
            }
            readGenotypes = false;
            gctb.inputIndInfo(data, opt.bedFile, opt.phenotypeFile, opt.keepIndFile, opt.keepIndMax,
                               opt.mphen, opt.covariateFile, opt.randomCovariateFile, opt.residualDiagFile);
            gctb.inputSnpInfo(data, opt.bedFile, opt.includeSnpFile, opt.excludeSnpFile, opt.excludeRegionFile, opt.includeChr, opt.excludeAmbiguousSNP, opt.skeletonSnpFile, opt.geneticMapFile, opt.ldBlockInfoFile, opt.includeBlock, opt.annotationFile, opt.transpose, opt.continuousAnnoFile, opt.flank, opt.eQTLFile, opt.mafmin, opt.mafmax, opt.noscale, readGenotypes);
            
            Model *model = gctb.buildModel(data, opt, opt.bedFile, "", opt.bayesType, opt.windowWidth,
                                            opt.heritability, opt.propVarRandom, opt.pi, opt.piAlpha, opt.piBeta, opt.estimatePi, opt.noscale, opt.pis, opt.piPar, opt.gamma, opt.estimateSigmaSq, opt.phi, opt.kappa,
                                            opt.algorithm, opt.snpFittedPerWindow, opt.varS, opt.S, opt.overdispersion, opt.estimatePS, opt.icrsq, opt.spouseCorrelation, opt.diagnosticMode, opt.hsqPercModel, opt.perSnpGV, opt.robustMode, opt.nDistAuto, opt.estimateRsqEnrich);
            vector<McmcSamples*> mcmcSampleVec = gctb.runMcmc(*model, 1, opt.chainLength, opt.burnin, opt.thin,
                                                               opt.outputFreq, opt.title, opt.writeBinPosterior, opt.writeTxtPosterior);
            //gctb.saveMcmcSamples(mcmcSampleVec, opt.title);
            gctb.clearGenotypes(data);
            if (opt.outputResults) gctb.outputResults(data, mcmcSampleVec, opt.bayesType, opt.noscale, opt.title);
        }
        else if (opt.analysisType == "LDmatrix") {
            readGenotypes = false;
            if (!opt.plinkLDtxtfile.empty() || !opt.plinkLDbinfile.empty()) {  // read from PLINK output file
                data.readPlinkAFfile(opt.plinkAFfile);
                if (!opt.plinkLDtxtfile.empty()) data.readPlinkLDtxtfile(opt.plinkLDtxtfile);
                if (!opt.plinkLDbinfile.empty()) data.readPlinkLDbinfile(opt.plinkLDbinfile);
                data.outputLDmatrix("full", opt.title, opt.writeLdmTxt);
            }
            else if (opt.ldmatrixFile.empty()) { // make LD matrix from genotypes
                gctb.inputIndInfo(data, opt.bedFile, opt.bedFile + ".fam", opt.keepIndFile, opt.keepIndMax,
                                  opt.mphen, opt.covariateFile, opt.randomCovariateFile, opt.residualDiagFile);
                gctb.inputSnpInfo(data, opt.bedFile, opt.includeSnpFile, opt.excludeSnpFile, opt.excludeRegionFile, opt.includeChr, opt.excludeAmbiguousSNP, opt.skeletonSnpFile, opt.geneticMapFile, opt.ldBlockInfoFile, opt.includeBlock, opt.annotationFile, opt.transpose, opt.continuousAnnoFile, opt.flank, opt.eQTLFile, opt.mafmin, opt.mafmax, opt.noscale, readGenotypes);
                if (opt.outLDmatType == "shrunk") {
                    data.makeshrunkLDmatrix(opt.bedFile + ".bed", opt.outLDmatType, opt.snpRange, opt.title, opt.writeLdmTxt, opt.effpopNE, opt.cutOff, opt.genMapN);
                } else if (opt.outLDmatType == "block") {
                    data.makeBlockLDmatrix(opt.bedFile + ".bed", opt.outLDmatType, opt.includeBlock, opt.title, opt.writeLdmTxt);
                }
                else {
                    string snpRange = opt.snpRange;
                    if(!opt.partParam.empty()){
                        snpRange = data.partLDMatrix(opt.partParam, opt.title, opt.outLDmatType);
                    }
                    data.makeLDmatrix(opt.bedFile + ".bed", opt.outLDmatType, opt.chisqThreshold, opt.LDthreshold, opt.windowWidth, snpRange, opt.title, opt.writeLdmTxt);
                }
            }
//            else if (opt.ldmatrixFile.empty() != 1 && opt.outLDmatType == "shrunk" || opt.outLDmatType == "sparseshrunk") { // make shrunk LD matrix from other LDM
//                gctb.inputSnpInfo(data, opt.includeSnpFile, opt.excludeSnpFile, "", opt.ldmatrixFile, opt.includeChr, opt.multiLDmat, opt.geneticMapFile);
//                data.resizeLDmatrix(opt.outLDmatType, opt.chisqThreshold, opt.windowWidth, opt.LDthreshold, opt.effpopNE, opt.cutOff, opt.genMapN);
//                data.outputLDmatrix(opt.outLDmatType, opt.title);
//            }
//            else if (opt.ldmatrixFile.empty() == 1 && opt.outLDmatType != "shrunk") { // make LD matrix from genotypes
//                gctb.inputIndInfo(data, opt.bedFile, opt.bedFile + ".fam", opt.keepIndFile, opt.keepIndMax,
//                                  opt.mphen, opt.covariateFile);
//                gctb.inputSnpInfo(data, opt.bedFile, opt.includeSnpFile, opt.excludeSnpFile, opt.includeChr, readGenotypes);
//                data.makeLDmatrix(opt.bedFile + ".bed", opt.outLDmatType, opt.chisqThreshold, opt.LDthreshold, opt.windowWidth, opt.snpRange, opt.title);
//            }
            else { // manipulate an existing LD matrix or merge existing LD matrices
                if (opt.mergeLdm) {
                    data.mergeLdmInfo(opt.outLDmatType, opt.ldmatrixFile, true);
                }
                else if (opt.directPrune) {
                    data.directPruneLDmatrix(opt.ldmatrixFile, opt.outLDmatType, opt.chisqThreshold, opt.title, opt.writeLdmTxt);
                }
                else if (opt.jackknife) {
                    readGenotypes = true;
                    gctb.inputIndInfo(data, opt.bedFile, opt.bedFile + ".fam", opt.keepIndFile, opt.keepIndMax, opt.mphen, opt.covariateFile, opt.randomCovariateFile, opt.residualDiagFile);
                    gctb.inputSnpInfo(data, opt.bedFile, opt.includeSnpFile, opt.excludeSnpFile, opt.excludeRegionFile, opt.includeChr, opt.excludeAmbiguousSNP, opt.skeletonSnpFile, opt.geneticMapFile, opt.ldBlockInfoFile, opt.includeBlock, opt.annotationFile, opt.transpose, opt.continuousAnnoFile, opt.flank, opt.eQTLFile, opt.mafmin, opt.mafmax, opt.noscale, readGenotypes);
                    data.jackknifeLDmatrix(opt.ldmatrixFile, opt.outLDmatType, opt.title, opt.writeLdmTxt);
                }
                else if (opt.binSnp) {
                    gctb.inputSnpInfo(data, opt.includeSnpFile, opt.excludeSnpFile, opt.excludeRegionFile, "", opt.ldmatrixFile, opt.includeChr, opt.excludeAmbiguousSNP, opt.skeletonSnpFile, opt.geneticMapFile, opt.genMapN, opt.annotationFile, opt.transpose, opt.continuousAnnoFile, opt.flank, opt.eQTLFile, opt.ldscoreFile, opt.windowFile, opt.multiLDmat, opt.excludeMHC, opt.afDiff, opt.mafmin, opt.mafmax, opt.pValueThreshold, opt.rsqThreshold, opt.sampleOverlap, opt.imputeN, opt.noscale, opt.binSnp, opt.readLdmTxt);
                    data.binSnpByLDrsq(opt.rsqThreshold, opt.title);
                }
                else if (opt.outLDmatType == "block") { // resize block matrix
                    if (!opt.includeSnpFile.empty()) {
                        data.resizeBlockLDmatrix(opt.ldmatrixFile, opt.outLDmatType, opt.includeSnpFile, opt.title, opt.writeLdmTxt);
                    }
                    else if (opt.writeLdmTxt) {
                        data.outputBlockLDmatrixTxt(opt.ldmatrixFile, opt.includeBlock);
                    }
                }
                else if (opt.outLDmatType == "blockTri") { // only save the lower triangular matrix
                    data.convertToBlockTriangularMatrix(opt.ldmatrixFile, opt.writeLdmTxt, opt.title);
                }
                else if (opt.outLDmatType == "blockSparse") {
                    if (opt.includeSnpFile.empty()) {  // make block sparse LD matrix from block full LD matrix
                        data.readBlockLDmatrixAndMakeItSparse(opt.ldmatrixFile, opt.includeBlock, opt.chisqThreshold, opt.writeLdmTxt);
                    } else {  // resize block sparse matrix
                        data.resizeBlockLDmatrix(opt.ldmatrixFile, opt.outLDmatType, opt.includeSnpFile, opt.title, opt.writeLdmTxt);
                    }
                }
                else {  // resize existing non-block LD matrix
                    gctb.inputSnpInfo(data, opt.includeSnpFile, opt.excludeSnpFile, opt.excludeRegionFile, "", opt.ldmatrixFile, opt.includeChr, opt.excludeAmbiguousSNP, opt.skeletonSnpFile, opt.geneticMapFile, opt.genMapN, opt.annotationFile, opt.transpose, opt.continuousAnnoFile, opt.flank, opt.eQTLFile, opt.ldscoreFile, opt.windowFile, opt.multiLDmat, opt.excludeMHC, opt.afDiff, opt.mafmin, opt.mafmax, opt.pValueThreshold, opt.rsqThreshold, opt.sampleOverlap, opt.imputeN, opt.noscale, opt.binSnp, opt.readLdmTxt);
                    data.resizeLDmatrix(opt.outLDmatType, opt.chisqThreshold, opt.windowWidth, opt.LDthreshold, opt.effpopNE, opt.cutOff, opt.genMapN);
                    data.outputLDmatrix(opt.outLDmatType, opt.title, opt.writeLdmTxt);
                }
            }
        }
        else if (opt.analysisType == "LDmatrixEigen") {
            readGenotypes = false;
            if (opt.eigenMatrixFile.empty()) { // perform eigen decomposition for the blocked LD matrices
                if (!opt.gwasSummaryFile.empty()) { // match LD ref SNPs with GWAS SNPs
                    gctb.inputSnpInfo(data, opt.includeSnpFile, opt.excludeSnpFile, opt.excludeRegionFile,
                                      opt.gwasSummaryFile, opt.ldmatrixFile, opt.ldBlockInfoFile,
                                      opt.includeChr, opt.excludeAmbiguousSNP,
                                      opt.annotationFile, opt.transpose,
                                      opt.continuousAnnoFile, opt.flank, opt.eQTLFile, opt.ldscoreFile,
                                      opt.eigenCutoff.maxCoeff(), opt.excludeMHC,
                                      opt.afDiff, opt.mafmin, opt.mafmax, opt.pValueThreshold, opt.rsqThreshold,
                                      opt.sampleOverlap, opt.imputeN, opt.noscale, opt.readLdmTxt, opt.imputeSummary, opt.includeBlock, opt.skipSnpFile, false);
                    data.resizeBlockLDmatrixAndDoEigenDecomposition(opt.ldmatrixFile, opt.eigenCutoff.maxCoeff(), 0.5, opt.title, opt.writeLdmTxt);
                } else {
                    //gctb.inputSnpInfo(data, opt.bedFile, opt.includeSnpFile, opt.excludeSnpFile, opt.excludeRegionFile, opt.includeChr, opt.excludeAmbiguousSNP, opt.skeletonSnpFile, opt.geneticMapFile, opt.annotationFile, opt.transpose, opt.continuousAnnoFile, opt.flank, opt.eQTLFile, opt.mafmin, opt.mafmax, opt.noscale, readGenotypes);
                    //data.getEigenDataFromFullLDM(opt.title, opt.eigenCutoff);
                    data.readBlockLDmatrixAndDoEigenDecomposition(opt.ldmatrixFile, opt.includeBlock, opt.eigenCutoff.maxCoeff(), opt.writeLdmTxt);
                }
            }
            else { // merge existing eigen matrices
                //gctb.inputSnpInfo(data, opt.includeSnpFile, opt.excludeSnpFile, opt.excludeRegionFile, "", opt.eigenMatrixFile, opt.ldBlockInfoFile, opt.includeChr, opt.excludeAmbiguousSNP, opt.annotationFile, opt.transpose, opt.continuousAnnoFile, opt.flank, opt.eQTLFile, opt.ldscoreFile, opt.eigenCutoff, opt.excludeMHC, opt.afDiff, opt.mafmin, opt.mafmax, opt.pValueThreshold, opt.rsqThreshold, opt.sampleOverlap, opt.imputeN, opt.noscale, opt.readLdmTxt);
            }
        }
        else if (opt.analysisType == "sparseLDmatrixEigen") {
            readGenotypes = false;
            data.readSparseBlockLDmatrixAndDoEigenDecomposition(opt.ldmatrixFile, opt.includeBlock, opt.eigenCutoff.maxCoeff(), opt.writeLdmTxt);
        }
        else if (opt.analysisType == "ImputeSumStats") {
            readGenotypes = false;
            if (opt.eigenMatrixFile.empty()) {
                throw("Error: --impute-summary requires the results of eigen-decomposition of LD matrices as input.");
            }
            gctb.inputSnpInfo(data, opt.includeSnpFile, opt.excludeSnpFile, opt.excludeRegionFile,
                              opt.gwasSummaryFile, opt.eigenMatrixFile, opt.ldBlockInfoFile,
                              opt.includeChr, opt.excludeAmbiguousSNP,
                              opt.annotationFile, opt.transpose,
                              opt.continuousAnnoFile, opt.flank, opt.eQTLFile, opt.ldscoreFile,
                              opt.eigenCutoff.maxCoeff(), opt.excludeMHC,
                              opt.afDiff, opt.mafmin, opt.mafmax, opt.pValueThreshold, opt.rsqThreshold,
                              opt.sampleOverlap, opt.imputeN, opt.noscale, opt.readLdmTxt, opt.imputeSummary, opt.includeBlock, opt.skipSnpFile);
        }
        else if (opt.analysisType == "MergeGwasSummary") {
            if (opt.outLDmatType == "block") {
                data.mergeBlockGwasSummary(opt.gwasSummaryFile, opt.title);
            }
        }
        else if (opt.analysisType == "Convert") {
            if (!opt.eigenMatrixFile.empty()) {
                data.readEigenMatrix(opt.eigenMatrixFile, opt.eigenCutoff.maxCoeff(), false, false, ".", opt.eigenMatrixQuantBits, opt.eigenMatrixQ8Entropy, opt.eigenMatrixQSnpColumn, opt.eigenMatrixUTranspose);
                data.inputMatchedSnpResults(opt.snpResFile);
                data.convert(opt.eigenMatrixFile, opt.includeSnpFile, opt.title, opt.eigenMatrixQuantBits, opt.eigenMatrixQ8Entropy, opt.eigenMatrixQSnpColumn, opt.eigenMatrixUTranspose);
            }
        }
        else if (opt.analysisType == "GetLD") {
            if (!opt.eigenMatrixFile.empty()) {
                data.readEigenMatrix(opt.eigenMatrixFile, opt.eigenCutoff.maxCoeff(), false, false, ".", opt.eigenMatrixQuantBits, opt.eigenMatrixQ8Entropy, opt.eigenMatrixQSnpColumn, opt.eigenMatrixUTranspose);
                data.getLDfromEigenMatrix(opt.eigenMatrixFile, opt.rsqThreshold, opt.title, opt.eigenMatrixQuantBits, opt.eigenMatrixQ8Entropy, opt.eigenMatrixQSnpColumn, opt.eigenMatrixUTranspose);
            }
        }
        else if (opt.analysisType == "GetLDfriends") {
            if (opt.rsqThreshold == 1.0) {
                opt.rsqThreshold = 0.5;  // default value
            }
            data.getLDfriends(opt.pairwiseLDfile, opt.rsqThreshold, opt.title);
        }
        else if (opt.analysisType == "SBayes" || opt.analysisType == "GWFM") {
            if (!opt.ldmatrixFile.empty()) {
                gctb.inputSnpInfo(data, opt.includeSnpFile, opt.excludeSnpFile, opt.excludeRegionFile, opt.gwasSummaryFile, opt.ldmatrixFile, opt.includeChr, opt.excludeAmbiguousSNP, opt.skeletonSnpFile, opt.geneticMapFile, opt.genMapN, opt.annotationFile, opt.transpose, opt.continuousAnnoFile, opt.flank, opt.eQTLFile, opt.ldscoreFile, opt.windowFile, opt.multiLDmat, opt.excludeMHC, opt.afDiff, opt.mafmin, opt.mafmax, opt.pValueThreshold, opt.rsqThreshold, opt.sampleOverlap, opt.imputeN, opt.noscale, opt.binSnp, opt.readLdmTxt);
            } else if (!opt.eigenMatrixFile.empty()) {  // low-rank model
                data.mergeLdmInfo("block", opt.eigenMatrixFile, false); // if each block has its own .info file, then merge them
                gctb.inputSnpInfo(data, opt.includeSnpFile, opt.excludeSnpFile, opt.excludeRegionFile,
                                  opt.gwasSummaryFile, opt.eigenMatrixFile, opt.ldBlockInfoFile,
                                  opt.includeChr, opt.excludeAmbiguousSNP,
                                  opt.annotationFile, opt.transpose,
                                  opt.continuousAnnoFile, opt.flank, opt.eQTLFile, opt.ldscoreFile,
                                  opt.eigenCutoff.maxCoeff(), opt.excludeMHC,
                                  opt.afDiff, opt.mafmin, opt.mafmax, opt.pValueThreshold, opt.rsqThreshold,
                                  opt.sampleOverlap, opt.imputeN, opt.noscale, opt.readLdmTxt, opt.imputeSummary, opt.includeBlock, opt.skipSnpFile);
               if (sbayesrEarlyStop && opt.analysisType == "SBayes" && opt.bayesType == "R") {
                    cout << "\nStopping before eigen-cutoff tuning/model selection/build and MCMC start for SBayesR as requested." << endl;
                    throw earlyStopToken;
                }
                if (opt.analysisType == "GWFM") {
                    data.inputPairwiseLD(opt.eigenMatrixFile+"/"+opt.pairwiseLDfile, 0.95);  // for TGS sampling
                }
                float bestEigenCutoff = opt.eigenCutoff.size() > 1 ? gctb.tuneEigenCutoff(data, opt) : opt.eigenCutoff[0];
                data.readEigenMatrixBinaryFileAndMakeWandQ(opt.eigenMatrixFile, bestEigenCutoff, data.gwasEffectInBlock, data.nGWASblock, opt.noscale, false, opt.eigenMatrixQuantBits, opt.eigenMatrixQ8Entropy, opt.eigenMatrixQSnpColumn, opt.eigenMatrixUTranspose);
                if (opt.writeWandQ) data.outputWandQ("w_and_Q");
                //data.readEigenMatrixBinaryFile(opt.eigenMatrixFile, bestEigenCutoff);
                //data.constructWandQ(data.gwasEffectInBlock, data.numKeptInds);
            } else {
                gctb.inputSnpInfo(data, opt.bedFile, opt.gwasSummaryFile, opt.afDiff, opt.mafmin, opt.mafmax, opt.pValueThreshold, opt.sampleOverlap, opt.imputeN, opt.noscale);
            }
            
            data.label = opt.title;
//            if (opt.numChains > 1) {
//                vector<McmcSamples*> mcmcSampleVec = gctb.multi_chain_mcmc(data, opt.bayesType, opt.windowWidth, opt.heritability, opt.propVarRandom, opt.pi, opt.piAlpha, opt.piBeta, opt.estimatePi, opt.pis, opt.gamma, opt.phi, opt.kappa, opt.algorithm, opt.snpFittedPerWindow, opt.varS, opt.S, opt.overdispersion, opt.estimatePS, opt.icrsq, opt.spouseCorrelation, opt.diagnosticMode, opt.robustMode, opt.numChains, opt.chainLength, opt.burnin, opt.thin, opt.outputFreq, opt.title, opt.writeBinPosterior, opt.writeTxtPosterior);
//                if (opt.outputResults) gctb.outputResults(data, mcmcSampleVec, opt.bayesType, opt.noscale, opt.title);
//            } else {
             
            if (sbayesrEarlyStop && opt.analysisType == "SBayes" && opt.bayesType == "R") {
                cout << "\nStopping before model selection/build and MCMC start for SBayesR as requested." << endl;
                throw earlyStopToken;
            }
            
            if (opt.nDistAuto) gctb.findBestFitModel(data, opt);
            
            Model *model = gctb.buildModel(data, opt, opt.bedFile, opt.gwasSummaryFile, opt.bayesType, opt.windowWidth,
                                           opt.heritability, opt.propVarRandom, opt.pi, opt.piAlpha, opt.piBeta, opt.estimatePi, opt.noscale, opt.pis, opt.piPar, opt.gamma, opt.estimateSigmaSq, opt.phi, opt.kappa,
                                           opt.algorithm, opt.snpFittedPerWindow, opt.varS, opt.S, opt.overdispersion, opt.estimatePS, opt.icrsq, opt.spouseCorrelation, opt.diagnosticMode, opt.hsqPercModel, opt.perSnpGV, opt.robustMode, opt.nDistAuto, opt.estimateRsqEnrich);
            
//            vector<McmcSamples*> mcmcSampleVec = gctb.runMcmc(*model, opt.numChains, opt.chainLength, opt.burnin, opt.thin,
//                                                              opt.outputFreq, opt.title, opt.writeBinPosterior, opt.writeTxtPosterior);

            MCMC *mcmc = new MCMC();
            if (sbayesrEarlyStop && opt.analysisType == "SBayes" && opt.bayesType == "R") {
    cout << "\nStopping after MCMC object creation and before MCMC run for SBayesR as requested." << endl;
    delete mcmc;
    delete model;
    mcmc = nullptr;
    model = nullptr;
    throw earlyStopToken;
}
            vector<McmcSamples*> mcmcSampleVec = mcmc->run(*model, opt.numChains, opt.chainLength, opt.burnin, opt.thin, true,
                                                           opt.outputFreq, opt.title, opt.writeBinPosterior, opt.writeTxtPosterior);

            if (mcmc->action == mcmc->restart_and_use_robust_model) {
                cout << "\nRestarting MCMC with a more robust parameterisation for SBayes" << opt.bayesType << " ..." << endl;
                cout << "Please refer to our website (https://cnsgenomics.com/software/gctb) for more information." << endl;
                opt.robustMode = true;
                model = gctb.buildModel(data, opt, opt.bedFile, opt.gwasSummaryFile, opt.bayesType, opt.windowWidth,
                                        opt.heritability, opt.propVarRandom, opt.pi, opt.piAlpha, opt.piBeta, opt.estimatePi, opt.noscale, opt.pis, opt.piPar, opt.gamma, opt.estimateSigmaSq, opt.phi, opt.kappa,
                                        opt.algorithm, opt.snpFittedPerWindow, opt.varS, opt.S, opt.overdispersion, opt.estimatePS, opt.icrsq, opt.spouseCorrelation, opt.diagnosticMode, opt.hsqPercModel, opt.perSnpGV, opt.robustMode, opt.nDistAuto, opt.estimateRsqEnrich);
                mcmc = new MCMC();
                vector<McmcSamples*> mcmcSampleVec = mcmc->run(*model, opt.numChains, opt.chainLength, opt.burnin, opt.thin, true,
                                                               opt.outputFreq, opt.title, opt.writeBinPosterior, opt.writeTxtPosterior);
            }
            
            if (opt.outputResults) gctb.outputResults(data, mcmcSampleVec, opt.bayesType, opt.noscale, opt.title);
            
            if (opt.analysisType == "GWFM") {
                McmcSamples *snpEffects = gctb.inputMcmcSamples(opt.title, "SnpEffects", "bin");
                data.inputNewSnpResults(opt.title + ".snpRes");
                data.inputPairwiseLD(opt.eigenMatrixFile+"/"+opt.pairwiseLDfile, opt.rsqThreshold);
                if (!opt.geneMapFile.empty()) data.readGeneMapFile(opt.geneMapFile, opt.flank, opt.genomeBuild);
                gctb.calcCredibleSets(data, *snpEffects, opt.pipThreshold, opt.pepThreshold, opt.title);
            }
        }
        else if (opt.analysisType == "CalcRsqEnrichment") {
            data.inputNewSnpResults(opt.snpResFile + ".snpRes");
            data.readAnnotationFile(opt.annotationFile, opt.transpose, true);
            data.setAnnoInfoVec();
            data.calcMarignalEnrichmentJackknife("rsq", opt.title);
            data.calcJointEnrichmentJackknifeLM("rsq", opt.title);
        }
        else if (opt.analysisType == "CalcHsqEnrichment") {
            data.inputNewSnpResults(opt.snpResFile + ".snpRes");
            data.readAnnotationFile(opt.annotationFile, opt.transpose, true);
            data.setAnnoInfoVec();
            data.calcMarignalEnrichmentJackknife("hsq", opt.title);
            data.calcJointEnrichmentJackknifeLM("hsq", opt.title);
        }
        else if (opt.analysisType == "CalcPipEnrichment") {
            data.inputNewSnpResults(opt.snpResFile + ".snpRes");
            data.readAnnotationFile(opt.annotationFile, opt.transpose, true);
            data.setAnnoInfoVec();
            data.calcMarignalEnrichmentJackknife("pip", opt.title);
            data.calcJointEnrichmentJackknifeLM("pip", opt.title);
        }
        else if (opt.analysisType == "CalcRsqEnrichmentPart") {
            data.inputNewSnpResults(opt.snpResFile + ".snpRes");
            data.readAnnotationFile(opt.annotationFile, opt.transpose, true);
            data.setAnnoInfoVec();
            data.calcMarignalEnrichmentJackknife("rsq", opt.title);
            data.calcJointEnrichmentJackknife("rsq", opt.title);
        }
        else if (opt.analysisType == "CalcHsqEnrichmentPart") {
            data.inputNewSnpResults(opt.snpResFile + ".snpRes");
            data.readAnnotationFile(opt.annotationFile, opt.transpose, true);
            data.setAnnoInfoVec();
            data.calcMarignalEnrichmentJackknife("hsq", opt.title);
            data.calcJointEnrichmentJackknife("hsq", opt.title);
        }
        else if (opt.analysisType == "CalcPipEnrichmentPart") {
            data.inputNewSnpResults(opt.snpResFile + ".snpRes");
            data.readAnnotationFile(opt.annotationFile, opt.transpose, true);
            data.setAnnoInfoVec();
            data.calcMarignalEnrichmentJackknife("pip", opt.title);
            data.calcJointEnrichmentJackknife("pip", opt.title);
        }
        else if (opt.analysisType == "ConjugateGradient") {
            gctb.inputSnpInfo(data, opt.includeSnpFile, opt.excludeSnpFile, opt.excludeRegionFile, opt.gwasSummaryFile, opt.ldmatrixFile, opt.includeChr, opt.excludeAmbiguousSNP, opt.skeletonSnpFile, opt.geneticMapFile, opt.genMapN, opt.annotationFile, opt.transpose, opt.continuousAnnoFile, opt.flank, opt.eQTLFile, opt.ldscoreFile, opt.windowFile, opt.multiLDmat, opt.excludeMHC, opt.afDiff, opt.mafmin, opt.mafmax, opt.pValueThreshold, opt.rsqThreshold, opt.sampleOverlap, opt.imputeN, opt.noscale, opt.binSnp, opt.readLdmTxt);
            gctb.solveSnpEffectsByConjugateGradientMethod(data, opt.lambda, opt.title + ".snpRes");
        }
        else if (opt.analysisType == "Stratify") { // post hoc stratified analysis
            gctb.stratify(data, opt.ldmatrixFile, opt.multiLDmat, opt.geneticMapFile, opt.genMapN, opt.snpResFile, opt.mcmcSampleFile, opt.annotationFile, opt.transpose, opt.continuousAnnoFile, opt.flank, opt.eQTLFile, opt.gwasSummaryFile, opt.pValueThreshold, opt.imputeN, opt.title, opt.bayesType, opt.chainLength, opt.burnin, opt.thin, opt.outputFreq);
        }
        else if (opt.analysisType == "hsq") {
            if (opt.ldmatrixFile.empty()) {
                readGenotypes = true;
                gctb.inputIndInfo(data, opt.bedFile, opt.phenotypeFile, opt.keepIndFile, opt.keepIndMax,
                               opt.mphen, opt.covariateFile, opt.randomCovariateFile, opt.residualDiagFile);
                gctb.inputSnpInfo(data, opt.bedFile, opt.includeSnpFile, opt.excludeSnpFile, opt.excludeRegionFile, opt.includeChr, opt.excludeAmbiguousSNP, opt.skeletonSnpFile, opt.geneticMapFile, opt.ldBlockInfoFile, opt.includeBlock, opt.annotationFile, opt.transpose, opt.continuousAnnoFile, opt.flank, opt.eQTLFile, opt.mafmin, opt.mafmax, opt.noscale, readGenotypes);
            } else {
                gctb.inputSnpInfo(data, opt.includeSnpFile, opt.excludeSnpFile, opt.excludeRegionFile, opt.gwasSummaryFile, opt.ldmatrixFile, opt.includeChr, opt.excludeAmbiguousSNP, opt.skeletonSnpFile, opt.geneticMapFile, opt.genMapN, opt.annotationFile, opt.transpose, opt.continuousAnnoFile, opt.flank, opt.eQTLFile, opt.ldscoreFile, opt.windowFile, opt.multiLDmat, opt.excludeMHC, opt.afDiff, opt.mafmin, opt.mafmax, opt.pValueThreshold, opt.rsqThreshold, opt.sampleOverlap, opt.imputeN, opt.noscale, opt.binSnp, opt.readLdmTxt);
                if (data.sparseLDM) data.getZPZspmat();
                else data.getZPZmat();
            }
            McmcSamples *snpEffects = gctb.inputMcmcSamples(opt.mcmcSampleFile, "SnpEffects", "bin");
            McmcSamples *resVar = gctb.inputMcmcSamples(opt.mcmcSampleFile, "ResVar", "txt");
            gctb.estimateHsq(data, *snpEffects, *resVar, opt.title, opt.outputFreq);
        }
        else if (opt.analysisType == "Pi") {
            if (opt.ldmatrixFile.empty()) {
                readGenotypes = true;  // need this to calculate allele frequencies
                gctb.inputIndInfo(data, opt.bedFile, opt.phenotypeFile, opt.keepIndFile, opt.keepIndMax,
                               opt.mphen, opt.covariateFile, opt.randomCovariateFile, opt.residualDiagFile);
                gctb.inputSnpInfo(data, opt.bedFile, opt.includeSnpFile, opt.excludeSnpFile, opt.excludeRegionFile, opt.includeChr, opt.excludeAmbiguousSNP, opt.skeletonSnpFile, opt.geneticMapFile, opt.ldBlockInfoFile, opt.includeBlock, opt.annotationFile, opt.transpose, opt.continuousAnnoFile, opt.flank, opt.eQTLFile, opt.mafmin, opt.mafmax, opt.noscale, readGenotypes);
            } else {
                gctb.inputSnpInfo(data, opt.includeSnpFile, opt.excludeSnpFile, opt.excludeRegionFile, opt.gwasSummaryFile, opt.ldmatrixFile, opt.includeChr, opt.excludeAmbiguousSNP, opt.skeletonSnpFile, opt.geneticMapFile, opt.genMapN, opt.annotationFile, opt.transpose, opt.continuousAnnoFile, opt.flank, opt.eQTLFile, opt.ldscoreFile, opt.windowFile, opt.multiLDmat, opt.excludeMHC, opt.afDiff, opt.mafmin, opt.mafmax, opt.pValueThreshold, opt.rsqThreshold, opt.sampleOverlap, opt.imputeN, opt.noscale, opt.binSnp, opt.readLdmTxt);
                //if (data.sparseLDM) data.getZPZspmat();
                //else data.getZPZmat();
            }
            McmcSamples *snpEffects = gctb.inputMcmcSamples(opt.mcmcSampleFile, "SnpEffects", "bin");
            McmcSamples *genVar = gctb.inputMcmcSamples(opt.mcmcSampleFile, "GenVar", "txt");
            gctb.estimatePi(data, *snpEffects, *genVar, opt.title, opt.outputFreq);
        }
        else if (opt.analysisType == "WindowPIP") {
            McmcSamples *snpEffects = gctb.inputMcmcSamples(opt.mcmcSampleFile, "SnpEffects", "bin");
            gctb.getWindowPIP(data, *snpEffects, opt.mcmcSampleFile + ".snpRes", opt.windowWidth, 0.5*opt.windowWidth, opt.title);
        }
        else if (opt.analysisType == "CS") {
//            if (!opt.eigenMatrixFile.empty()) {
//                McmcSamples *snpEffects = gctb.inputMcmcSamples(opt.mcmcSampleFile, "SnpEffects", "bin");
//                gctb.calcCredibleSets(data, opt.mcmcSampleFile + ".snpRes", *snpEffects, opt.eigenMatrixFile, opt.eigenCutoff.maxCoeff(), opt.pipThreshold, opt.title);
//            } 
            if (!opt.pairwiseLDfile.empty()) {
                McmcSamples *snpEffects = gctb.inputMcmcSamples(opt.mcmcSampleFile, "SnpEffects", "bin");
                data.inputNewSnpResults(opt.mcmcSampleFile + ".snpRes");
                data.inputPairwiseLD(opt.eigenMatrixFile+"/"+opt.pairwiseLDfile, opt.rsqThreshold);
                if (!opt.geneMapFile.empty()) data.readGeneMapFile(opt.geneMapFile, opt.flank, opt.genomeBuild);
                gctb.calcCredibleSets(data, *snpEffects, opt.pipThreshold, opt.pepThreshold, opt.title);
            }
            else if (!opt.ldfriendFile.empty()) {
                McmcSamples *snpEffects = gctb.inputMcmcSamples(opt.mcmcSampleFile, "SnpEffects", "bin");
                data.inputNewSnpResults(opt.mcmcSampleFile + ".snpRes");
                data.inputLDfriends(opt.eigenMatrixFile+"/"+opt.ldfriendFile);
                if (!opt.geneMapFile.empty()) data.readGeneMapFile(opt.geneMapFile, opt.flank, opt.genomeBuild);
                gctb.calcCredibleSets(data, *snpEffects, opt.pipThreshold, opt.pepThreshold, opt.title);
            } else {
                int windowWidth = 100000;
                if (opt.windowWidth) windowWidth = opt.windowWidth;
                McmcSamples *snpEffects = gctb.inputMcmcSamples(opt.mcmcSampleFile, "SnpEffects", "bin");
                gctb.calcCredibleSets(data, opt.mcmcSampleFile + ".snpRes", *snpEffects, opt.pipThreshold, opt.pepThreshold, windowWidth, opt.title);
            }
        }
        else if (opt.analysisType == "Bin2Txt") {
            McmcSamples *mcmcSamples = gctb.inputMcmcSamples(opt.mcmcSampleFile, opt.label, "bin");
            mcmcSamples->writeMatSpTxt(opt.mcmcSampleFile);
        }
        else if (opt.analysisType == "Print") {
            if (!opt.eigenMatrixFile.empty()) {
                data.readEigenMatrix(opt.eigenMatrixFile, opt.eigenCutoff.maxCoeff(), true, true, opt.title, opt.eigenMatrixQuantBits, opt.eigenMatrixQ8Entropy, opt.eigenMatrixQSnpColumn, opt.eigenMatrixUTranspose);
            }
        }
        else if (opt.analysisType == "Predict") {
            readGenotypes = true;
            gctb.inputIndInfo(data, opt.bedFile, opt.phenotypeFile, opt.keepIndFile, opt.keepIndMax,
                               opt.mphen, opt.covariateFile, opt.randomCovariateFile, opt.residualDiagFile);
            gctb.inputSnpInfo(data, opt.bedFile, opt.includeSnpFile, opt.excludeSnpFile, opt.excludeRegionFile, opt.includeChr, opt.excludeAmbiguousSNP, opt.skeletonSnpFile, opt.geneticMapFile, opt.ldBlockInfoFile, opt.includeBlock, opt.annotationFile, opt.transpose, opt.continuousAnnoFile, opt.flank, opt.eQTLFile, opt.mafmin, opt.mafmax, opt.noscale, readGenotypes);
            
            data.inputMatchedSnpResults(opt.snpResFile);
            gctb.predict(data, opt.title);
        }
        else if (opt.analysisType == "Summarize") {  // ad hoc method for producing summary from binary MCMC samples of SNP effects
            readGenotypes = true;
            gctb.inputIndInfo(data, opt.bedFile, opt.phenotypeFile, opt.keepIndFile, opt.keepIndMax,
                               opt.mphen, opt.covariateFile, opt.randomCovariateFile, opt.residualDiagFile);
            gctb.inputSnpInfo(data, opt.bedFile, opt.includeSnpFile, opt.excludeSnpFile, opt.excludeRegionFile, opt.includeChr, opt.excludeAmbiguousSNP, opt.skeletonSnpFile, opt.geneticMapFile, opt.ldBlockInfoFile, opt.includeBlock, opt.annotationFile, opt.transpose, opt.continuousAnnoFile, opt.flank, opt.eQTLFile, opt.mafmin, opt.mafmax, opt.noscale, readGenotypes);
            gctb.clearGenotypes(data);
            McmcSamples *snpEffects = gctb.inputMcmcSamples(opt.mcmcSampleFile, "SnpEffects", "bin");
            data.summarizeSnpResults(snpEffects->datMatSp, opt.title + ".snpRes");
        }
        else if (opt.analysisType == "XCI") {  // ad hoc method for X chromosome inactivation project
            XCI xci;
            readGenotypes = true;
            xci.inputIndInfo(data, opt.bedFile, opt.phenotypeFile, opt.keepIndFile, opt.keepIndMax,
                             opt.mphen, opt.covariateFile);
            xci.inputSnpInfo(data, opt.bedFile, opt.includeSnpFile, opt.excludeSnpFile, opt.includeChr, opt.annotationFile, opt.windowFile, readGenotypes);
            if (opt.simuMode) {
                xci.simu(data, opt.pi, opt.heritability, opt.piNDC, opt.piGxE, false, opt.title, opt.seed);  // ad hoc simulation to test BayesXCI method
            }
            else {
                if (opt.twoStageModel) {
                    // Stage 1: estimate NDC using female data only
                    Model *model = xci.buildModelStageOne(data, "C", opt.heritability, opt.pi, opt.piPar, opt.estimatePi, opt.piNDC, opt.piNDCpar, opt.estimatePiNDC);
                    vector<McmcSamples*> mcmcSampleVec = gctb.runMcmc(*model, 1, opt.chainLength, opt.burnin, opt.thin,
                                                                      opt.outputFreq, opt.title + ".stage1", opt.writeBinPosterior, opt.writeTxtPosterior);
                    gctb.saveMcmcSamples(mcmcSampleVec, opt.title + ".stage1");
                    gctb.outputResults(data, mcmcSampleVec, "C", true, opt.title + ".stage1");
                    xci.outputResults(data, mcmcSampleVec, "C", opt.title + ".stage1");
                    delete model;
                    // Stage 2: estimate GxS using both male and female data
                    model = xci.buildModelStageTwo(data, "Cgxs", opt.heritability, opt.pi, opt.piPar, opt.estimatePi, opt.piNDC, opt.piNDCpar, opt.estimatePiNDC, opt.title + ".stage1.snpRes", opt.piGxE, opt.estimatePiGxE);
                    mcmcSampleVec = gctb.runMcmc(*model, 1, opt.chainLength, opt.burnin, opt.thin,
                                                                      opt.outputFreq, opt.title + ".stage2", opt.writeBinPosterior, opt.writeTxtPosterior);
                    gctb.saveMcmcSamples(mcmcSampleVec, opt.title + ".stage2");
                    gctb.clearGenotypes(data);
                    gctb.outputResults(data, mcmcSampleVec, "Cgxs", true, opt.title + ".stage2");
                    xci.outputResults(data, mcmcSampleVec, "Cgxs", opt.title + ".stage2");
                    
                }
                else {
                    if (opt.numChains > 1) {  // multi chains
                        vector<McmcSamples*> mcmcSampleVec = xci.multi_chain_mcmc(data, opt.bayesType, opt.heritability, opt.pi, opt.piPar, opt.estimatePi, opt.piNDC, opt.piNDCpar, opt.estimatePiNDC, opt.piGxE, opt.estimatePiGxE, opt.numChains, opt.chainLength, opt.burnin, opt.thin, opt.outputFreq, opt.title, opt.writeBinPosterior, opt.writeTxtPosterior);
                        gctb.saveMcmcSamples(mcmcSampleVec, opt.title);
                        gctb.clearGenotypes(data);
                        gctb.outputResults(data, mcmcSampleVec, opt.bayesType, true, opt.title);
                        xci.outputResults(data, mcmcSampleVec, opt.bayesType, opt.title);
                    } else {
                        Model *model = xci.buildModel(data, opt.bayesType, opt.heritability, opt.pi, opt.piPar, opt.estimatePi, opt.piNDC, opt.piNDCpar, opt.estimatePiNDC, opt.piGxE, opt.estimatePiGxE, opt.windowWidth);
                        vector<McmcSamples*> mcmcSampleVec = gctb.runMcmc(*model, 1, opt.chainLength, opt.burnin, opt.thin,
                                                                          opt.outputFreq, opt.title, opt.writeBinPosterior, opt.writeTxtPosterior);
                        gctb.saveMcmcSamples(mcmcSampleVec, opt.title);
                        gctb.clearGenotypes(data);
                        gctb.outputResults(data, mcmcSampleVec, opt.bayesType, true, opt.title);
                        xci.outputResults(data, mcmcSampleVec, opt.bayesType, opt.title);
                    }
                }
            }
        }
        else if (opt.analysisType == "VGMAF") {  // ad hoc method for cumulative Vg against MAF to detect selection
            readGenotypes = true;
            gctb.inputIndInfo(data, opt.bedFile, opt.phenotypeFile, opt.keepIndFile, opt.keepIndMax,
                               opt.mphen, opt.covariateFile, opt.randomCovariateFile, opt.residualDiagFile);
            gctb.inputSnpInfo(data, opt.bedFile, opt.includeSnpFile, opt.excludeSnpFile, opt.excludeRegionFile, opt.includeChr, opt.excludeAmbiguousSNP, opt.skeletonSnpFile, opt.geneticMapFile, opt.ldBlockInfoFile, opt.includeBlock, opt.annotationFile, opt.transpose, opt.continuousAnnoFile, opt.flank, opt.eQTLFile, opt.mafmin, opt.mafmax, opt.noscale, readGenotypes);
            VGMAF vgmaf;
            if (opt.bayesType == "Simu") {
                vgmaf.simulate(data, opt.title);
            } else {
                McmcSamples *snpEffects = gctb.inputMcmcSamples(opt.mcmcSampleFile, "SnpEffects", "bin");
                vgmaf.compute(data, *snpEffects, opt.burnin, opt.thin, opt.title);
            }
        }
        
        else if (opt.analysisType == "OutputEffectSamples") { // for now an ad hoc method to output the MCMC SNP effect samples in text file
            McmcSamples *snpEffects = gctb.inputMcmcSamples(opt.mcmcSampleFile, "SnpEffects", "bin");
            data.outputSnpEffectSamples(snpEffects->datMatSp, opt.burnin, opt.outputFreq, opt.snpResFile, opt.title + ".snpEffectSamples");
        }
        
        else {
            throw(" Error: Wrong analysis type: " + opt.analysisType);
        }
    }
    catch (const std::string &err_msg) {
        if (err_msg != earlyStopToken) {
            cerr << "\n" << err_msg << endl;
        }
    }
    catch (const char *err_msg) {
        cerr << "\n" << err_msg << endl;
    }
    
    timer.getTime();
    
    cout << "\nAnalysis finished: " << timer.getDate();
    cout << "Computational time: "  << timer.format(timer.getElapse()) << endl;
#ifdef __linux__
    const int memPeakKB = getMemPeakKB();
    const int vmPeakKB = getVMPeakKB();
    if (memPeakKB >= 0 && vmPeakKB >= 0) {
        float vmem = roundf(1000.0f * vmPeakKB / 1024.0f / 1024.0f) / 1000.0f;
        float mem = roundf(1000.0f * memPeakKB / 1024.0f / 1024.0f) / 1000.0f;
        cout << fixed << setprecision(3)
             << "Peak memory: " << mem << " GB; Virtual memory: " << vmem << " GB." << endl;
    }
#endif

    return 0;
}
