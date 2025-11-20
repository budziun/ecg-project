import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TestSample, SignalData } from '../types';

interface ResultDisplayProps {
    result: TestSample;
    signal: SignalData | number[];
}

const ResultDisplay: React.FC<ResultDisplayProps> = ({ result, signal }) => {
    const [viewMode, setViewMode] = useState<'normalized' | 'raw'>('normalized');

    const signalData: SignalData = Array.isArray(signal)
        ? { normalized: signal, raw: [] }
        : signal;

    const activeSignal = viewMode === 'normalized'
        ? signalData.normalized
        : signalData.raw;

    let trimmedSignal = [...activeSignal];
    while (trimmedSignal.length > 0 && trimmedSignal[trimmedSignal.length - 1] === 0) {
        trimmedSignal.pop();
    }

    if (trimmedSignal.length < 20) {
        trimmedSignal = activeSignal;
    }

    const chartData = trimmedSignal.map((voltage, index) => ({
        time: index,
        voltage: Number(voltage.toFixed(4))
    }));

    const classColors: Record<string, { bg: string; text: string; bar: string }> = {
        'Normal': { bg: 'bg-green-50', text: 'text-green-700', bar: 'bg-green-500' },
        'Supraventricular': { bg: 'bg-yellow-50', text: 'text-yellow-700', bar: 'bg-yellow-500' },
        'Ventricular': { bg: 'bg-red-50', text: 'text-red-700', bar: 'bg-red-500' },
        'Fusion': { bg: 'bg-orange-50', text: 'text-orange-700', bar: 'bg-orange-500' },
        'Unknown': { bg: 'bg-gray-50', text: 'text-gray-700', bar: 'bg-gray-500' },
    };

    const colors = classColors[result.predicted_class] || classColors['Normal'];
    const trueColors = result.true_label ? classColors[result.true_label] || classColors['Normal'] : null;

    return (
        <div className="space-y-6">
            {/* ECG Signal Chart */}
            <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-semibold text-gray-800">📈 ECG Signal</h3>

                    <div className="flex gap-1 bg-gray-100 rounded-lg p-1">
                        <button
                            onClick={() => setViewMode('normalized')}
                            className={`px-3 py-1 text-sm rounded-md transition-colors ${
                                viewMode === 'normalized'
                                    ? 'bg-white text-blue-600 font-medium shadow-sm'
                                    : 'text-gray-600 hover:text-gray-800'
                            }`}
                        >
                            Normalized
                        </button>
                        <button
                            onClick={() => setViewMode('raw')}
                            className={`px-3 py-1 text-sm rounded-md transition-colors ${
                                viewMode === 'raw'
                                    ? 'bg-white text-blue-600 font-medium shadow-sm'
                                    : 'text-gray-600 hover:text-gray-800'
                            }`}
                            disabled={!signalData.raw || signalData.raw.length === 0}
                        >
                            Raw
                        </button>
                    </div>
                </div>

                <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="time" />
                        <YAxis />
                        <Tooltip />
                        <Line type="monotone" dataKey="voltage" stroke="#3b82f6" dot={false} />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            {/* Results Grid */}
            <div className="grid grid-cols-2 gap-6">
                {/* LEFT COLUMN - Model Prediction */}
                <div className={`${colors.bg} rounded-lg shadow p-6 space-y-4`}>
                    <div>
                        <p className="text-sm text-gray-600 mb-1">Model Prediction</p>
                        <p className={`text-4xl font-extrabold ${colors.text}`}>{result.predicted_class}</p>
                    </div>

                    <div>
                        <p className="text-sm text-gray-600 mb-2">Confidence Score</p>
                        <div className="w-full bg-gray-200 rounded-full h-3 mb-1">
                            <div
                                className={`${colors.bar} h-3 rounded-full`}
                                style={{ width: `${result.confidence * 100}%` }}
                            />
                        </div>
                        <p className="text-sm font-semibold text-gray-700">{(result.confidence * 100).toFixed(4)}%</p>
                    </div>

                    {result.is_uncertain && (
                        <div className="bg-yellow-100 border border-yellow-300 rounded p-3 text-sm text-yellow-800">
                            ⚠️ UNCERTAIN - May require medical review
                        </div>
                    )}

                    {/* TRUE LABEL */}
                    {result.true_label && trueColors && (
                        <div className="border-t-4 pt-6">
                            <h3 className="text-lg font-semibold text-gray-900 mb-4">True Label (MIT-BIH Database)</h3>

                            <div className="flex items-center justify-between mb-4">
                                <p className={`text-3xl font-bold ${trueColors.text}`}>
                                    {result.true_label}
                                </p>

                                <div className="text-right">
                                    {result.is_correct ? (
                                        <div className="flex flex-col items-center">
                                            <span className="text-4xl">✅</span>
                                            <span className="text-sm font-bold text-green-600 mt-1">True</span>
                                        </div>
                                    ) : (
                                        <div className="flex flex-col items-center">
                                            <span className="text-4xl">❌</span>
                                            <span className="text-sm font-bold text-red-600 mt-1">False</span>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* RIGHT COLUMN - Class Probabilities */}
                <div className="bg-white rounded-lg shadow p-6">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4">Class Probabilities</h3>
                    <div className="space-y-3">
                        {['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown'].map((cls) => {
                            const prob = result[cls as keyof TestSample] as number;
                            const pctDecimal = (prob * 100).toFixed(4);
                            const clsColor = classColors[cls];
                            return (
                                <div key={cls}>
                                    <div className="flex justify-between text-sm mb-1">
                                        <span className="font-medium">{cls}</span>
                                        <span className="text-gray-600">{pctDecimal}%</span>
                                    </div>
                                    <div className="w-full bg-gray-200 rounded-full h-2">
                                        <div
                                            className={`${clsColor.bar} h-2 rounded-full`}
                                            style={{ width: `${prob * 100}%` }}
                                        />
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ResultDisplay;
