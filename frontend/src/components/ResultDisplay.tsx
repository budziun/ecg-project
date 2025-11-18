import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TestSample } from '../types';

interface ResultDisplayProps {
    result: TestSample;
    signal: number[];
}

const ResultDisplay: React.FC<ResultDisplayProps> = ({ result, signal }) => {
    // ✅ Filtruj trailing zeros
    let trimmedSignal = [...signal];
    while (trimmedSignal.length > 0 && trimmedSignal[trimmedSignal.length - 1] === 0) {
        trimmedSignal.pop();
    }
    if (trimmedSignal.length < 20) {
        trimmedSignal = signal;
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
        <div className="space-y-8">
            {/* ECG Signal Chart - SMALLER HEIGHT */}
            <div className="bg-white rounded-lg shadow-lg p-6">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">📈 ECG Signal</h2>
                <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                        <XAxis dataKey="time" stroke="#666" />
                        <YAxis stroke="#666" />
                        <Tooltip />
                        <Line
                            type="monotone"
                            dataKey="voltage"
                            stroke="#0066FF"
                            dot={false}
                            isAnimationActive={false}
                            strokeWidth={2}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* LEFT COLUMN */}
                <div className={`${colors.bg} border-2 ${colors.text} rounded-lg p-8 shadow-lg`}>
                    <div className="mb-8 pb-6 border-b-2 border-opacity-20">
                        <h2 className="text-2xl font-bold text-gray-900">Model Prediction</h2>
                    </div>

                    <div className="mb-8">
                        <p className={`text-5xl font-bold mb-4 ${colors.text}`}>
                            {result.predicted_class}
                        </p>

                        <div>
                            <p className="text-sm opacity-80 mb-2">Confidence Score</p>
                            <div className="flex items-center space-x-3">
                                <div className="flex-1 bg-gray-300 rounded-full h-3">
                                    <div
                                        className={`${colors.bar} h-3 rounded-full transition-all`}
                                        style={{ width: `${result.confidence * 100}%` }}
                                    />
                                </div>
                                <span className="text-2xl font-bold font-mono">
                  {(result.confidence * 100).toFixed(4)}%
                </span>
                            </div>
                        </div>

                        <div className="mt-4">
                            {result.is_uncertain && (
                                <div className="p-3 bg-yellow-200 border border-yellow-600 rounded text-yellow-800 text-sm font-semibold">
                                    ⚠️ UNCERTAIN - May require medical review
                                </div>
                            )}
                        </div>
                    </div>

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

                {/* RIGHT COLUMN: Class Probabilities */}
                <div className="bg-white rounded-lg shadow-lg p-8 border-2 border-gray-200">
                    <h3 className="text-2xl font-bold text-gray-900 mb-6">Class Probabilities</h3>

                    <div className="space-y-5">
                        {['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown'].map((cls) => {
                            const prob = result[cls as keyof TestSample] as number;
                            const pctDecimal = (prob * 100).toFixed(4);
                            const clsColor = classColors[cls];

                            return (
                                <div key={cls}>
                                    <div className="flex justify-between mb-2">
                                        <span className="font-bold text-gray-800">{cls}</span>
                                        <span className="text-gray-700 font-mono font-bold">{pctDecimal}%</span>
                                    </div>
                                    <div className="w-full bg-gray-200 rounded-full h-4">
                                        <div
                                            className={`${clsColor.bar} h-4 rounded-full transition-all`}
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
