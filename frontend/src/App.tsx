import React, { useState } from 'react';
import { TestSample, SignalData } from './types';
import ResultDisplay from './components/ResultDisplay';
import TestSamples from './components/TestSamples';
import UploadECG from './components/UploadECG';
import AboutModal from './components/AboutModal';
import logo from './logo.png';

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
    const [signal, setSignal] = useState<SignalData>({ raw: [], normalized: [] });
    const [isAboutModalOpen, setIsAboutModalOpen] = useState(false);

    const isTestSample = (data: ResultType): data is TestSample => {
        return 'true_label' in data && 'is_correct' in data;
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-8 px-4">
            {/* HEADER */}
            <header className="max-w-7xl mx-auto mb-8">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        {logo && <img src={logo} alt="Logo" className="h-12 w-12 object-contain" />}
                        <div>
                            <h1 className="text-3xl font-bold text-gray-800">ECG Arrhythmia Classifier</h1>
                            <p className="text-sm text-gray-600">AI-powered heart rhythm analysis with MIT-BIH dataset</p>
                        </div>
                    </div>

                    {/* ✅ Przycisk About lub Back - w tym samym miejscu */}
                    {result ? (
                        <button
                            onClick={() => {
                                setResult(null);
                                setSignal({ raw: [], normalized: [] });
                            }}
                            className="px-6 py-3 bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold rounded-lg transition whitespace-nowrap ml-4"
                        >
                            ← Back
                        </button>
                    ) : (
                        <button
                            onClick={() => setIsAboutModalOpen(true)}
                            className="px-6 py-3 bg-blue-500 hover:bg-blue-600 text-white font-semibold rounded-lg transition whitespace-nowrap ml-4"
                        >
                            About Project
                        </button>
                    )}
                </div>
            </header>

            <div className="max-w-7xl mx-auto">
                {!result ? (
                    <div className="space-y-6">
                        {/* UPLOAD */}
                        <div className="bg-white rounded-xl shadow-lg p-6">
                            <UploadECG
                                onResult={(uploadedResult, uploadedSignal) => {
                                    setResult(uploadedResult);
                                    setSignal(uploadedSignal);
                                }}
                            />
                        </div>

                        {/* MIT-BIH TEST */}
                        <div className="bg-white rounded-xl shadow-lg p-6">
                            <h2 className="text-xl font-bold text-gray-800 mb-2 text-center">
                                MIT-BIH Test Data
                            </h2>
                            <p className="text-sm text-gray-600 mb-6 text-center">
                                Click button to load random ECG sample from MIT-BIH Arrhythmia Database
                            </p>
                            <TestSamples
                                onSelect={(sample, sig) => {
                                    setResult(sample);
                                    setSignal(sig);
                                }}
                            />
                        </div>
                    </div>
                ) : (
                    <div className="bg-white rounded-xl shadow-lg p-8">
                        {result && (signal.raw.length > 0 || signal.normalized.length > 0) && isTestSample(result) && (
                            <ResultDisplay result={result} signal={signal} />
                        )}
                        {result && (signal.raw.length > 0 || signal.normalized.length > 0) && !isTestSample(result) && (
                            <ResultDisplay result={result as any} signal={signal} />
                        )}
                    </div>
                )}
            </div>

            {/* ✅ About Modal */}
            <AboutModal
                isOpen={isAboutModalOpen}
                onClose={() => setIsAboutModalOpen(false)}
            />
        </div>
    );
}

export default App;
