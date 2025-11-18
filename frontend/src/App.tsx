import React, { useState } from 'react';
import { TestSample } from './types';
import ResultDisplay from './components/ResultDisplay';
import TestSamples from './components/TestSamples';
import UploadECG from './components/UploadECG';
import logo from './logo.png';

// UNION TYPE - dla TestSample LUB UploadResponse
interface UploadResponse {
    predicted_class: string;
    confidence: number;
    is_uncertain: boolean;
    Normal: number;
    Supraventricular: number;
    Ventricular: number;
    Fusion: number;
    Unknown: number;
    signal_for_plot: number[];
    signal_normalized: number[];
    threshold: number;
}

type ResultType = TestSample | UploadResponse;

function App() {
    const [result, setResult] = useState<ResultType | null>(null);
    const [signal, setSignal] = useState<number[]>([]);

    // TYPE GUARD - sprawdzenie czy to TestSample
    const isTestSample = (data: ResultType): data is TestSample => {
        return 'true_label' in data && 'is_correct' in data;
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
            {/* HEADER - WITH BACK BUTTON */}
            <div className="bg-white shadow">
                <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
                    {/* LEFT: Logo + Title */}
                    <div className="flex items-center gap-4">
                        {logo && <img src={logo} alt="ECG Logo" className="h-12" />}
                        <div>
                            <h1 className="text-3xl font-bold text-gray-900">ECG Arrhythmia Classifier</h1>
                            <p className="text-gray-600">AI-powered heart rhythm analysis with MIT-BIH dataset</p>
                        </div>
                    </div>

                    {/* RIGHT: Back Button (if results) */}
                    {result && (
                        <button
                            onClick={() => {
                                setResult(null);
                                setSignal([]);
                            }}
                            className="px-6 py-3 bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold rounded-lg transition whitespace-nowrap ml-4"
                        >
                            ← Back
                        </button>
                    )}
                </div>
            </div>

            {/* Main Content */}
            <div className="max-w-6xl mx-auto px-6 py-8">
                {!result ? (
                    <div>
                        {/* TEST SAMPLES SECTION */}
                        <div className="bg-white rounded-lg shadow-lg p-8 mb-8">
                            <div className="text-center mb-8">
                                <h2 className="text-2xl font-bold text-gray-900 mb-2">MIT-BIH Test Data</h2>
                                <p className="text-gray-600">Click button to load random ECG sample from MIT-BIH Arrhythmia Database</p>
                            </div>
                            <TestSamples
                                onSelect={(sample, sig) => {
                                    setResult(sample);
                                    setSignal(sig);
                                }}
                            />
                        </div>

                        {/* CSV UPLOAD SECTION */}
                        <div className="bg-white rounded-lg shadow-lg p-8">
                            <div className="text-center mb-4">
                                <h2 className="text-2xl font-bold text-gray-900 mb-2">Upload Your Own CSV File</h2>
                                <p className="text-gray-600">Upload an ECG CSV file for arrhythmia analysis</p>
                            </div>
                            <UploadECG
                                onResult={(uploadedResult, uploadedSignal) => {
                                    setResult(uploadedResult);
                                    setSignal(uploadedSignal);
                                }}
                            />
                        </div>
                    </div>
                ) : (
                    <div>
                        {/* RESULTS DISPLAY - SUPPORT BOTH TYPES */}
                        {result && signal.length > 0 && isTestSample(result) && (
                            <ResultDisplay result={result} signal={signal} />
                        )}
                        {/* PLACEHOLDER FOR UploadResponse - for future */}
                        {result && signal.length > 0 && !isTestSample(result) && (
                            <div className="bg-white rounded-lg shadow-lg p-8 text-center">
                                <p className="text-gray-600">CSV Upload result component coming soon...</p>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

export default App;
