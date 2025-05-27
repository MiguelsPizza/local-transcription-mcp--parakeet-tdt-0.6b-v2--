import { Alert, AlertDescription, AlertTitle } from "@workspace/ui/components/alert";
import { Button } from "@workspace/ui/components/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@workspace/ui/components/card";
import { Input } from "@workspace/ui/components/input";
import { Label } from "@workspace/ui/components/label";
import { Progress } from "@workspace/ui/components/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@workspace/ui/components/select";
import { Textarea } from "@workspace/ui/components/textarea";
import { type ChangeEvent, type FormEvent, useState } from "react";
import { transcribeAudio, type TranscriptionResponse } from "../lib/api";

export function TranscriptionForm() {
    const [file, setFile] = useState<File | null>(null);
    const [outputFormat, setOutputFormat] = useState<"wav" | "flac">("wav");
    const [includeTimestamps, setIncludeTimestamps] = useState(true);
    const [lineCharLimit, setLineCharLimit] = useState(80);
    const [segmentLengthMinutes, setSegmentLengthMinutes] = useState(5);

    const [transcriptionResult, setResult] = useState<TranscriptionResponse | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [progress, setProgress] = useState(0); // Simple progress based on loading state

    const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setResult(null); // Reset previous result
            setError(null); // Reset previous error
        }
    };

    const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        if (!file) {
            setError("Please select a file to transcribe.");
            return;
        }

        setIsLoading(true);
        setError(null);
        setResult(null);
        setProgress(30); // Initial progress indication

        const formData = new FormData();
        formData.append("file", file);
        formData.append("output_format", outputFormat);
        formData.append("include_timestamps", String(includeTimestamps));
        formData.append("line_character_limit", String(lineCharLimit));
        formData.append("segment_length_minutes", String(segmentLengthMinutes));

        try {
            setProgress(70); // Progress before API call
            const data = await transcribeAudio(formData);
            setResult(data);
            setProgress(100);
        } catch (err) {
            setError(err instanceof Error ? err.message : "An unknown transcription error occurred.");
            setProgress(0); // Reset progress on error
        } finally {
            setIsLoading(false);
            // Keep progress at 100 on success, or 0 on error. Reset if needed for retry.
        }
    };

    return (
        <Card className="w-full max-w-2xl mx-auto">
            <CardHeader>
                <CardTitle>Transcribe Audio/Video</CardTitle>
                <CardDescription>
                    Upload an audio or video file to transcribe it using the Parakeet TDT 0.6B V2 model.
                </CardDescription>
            </CardHeader>
            <form onSubmit={handleSubmit}>
                <CardContent className="space-y-6">
                    <div className="space-y-2">
                        <Label htmlFor="audio-file">Audio/Video File</Label>
                        <Input id="audio-file" type="file" onChange={handleFileChange} accept="audio/*,video/*" />
                        {file && <p className="text-sm text-muted-foreground">Selected: {file.name}</p>}
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-2">
                            <Label htmlFor="output-format">Intermediate Format</Label>
                            <Select value={outputFormat} onValueChange={(value: "wav" | "flac") => setOutputFormat(value)}>
                                <SelectTrigger id="output-format">
                                    <SelectValue placeholder="Select format" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="wav">WAV</SelectItem>
                                    <SelectItem value="flac">FLAC</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>

                        <div className="space-y-2">
                            <Label htmlFor="segment-length">Segment Length (minutes)</Label>
                            <Input
                                id="segment-length"
                                type="number"
                                value={segmentLengthMinutes}
                                onChange={(e) => setSegmentLengthMinutes(Math.max(1, Math.min(24, parseInt(e.target.value, 10) || 5)))}
                                min={1}
                                max={24}
                            />
                        </div>
                    </div>

                    <div className="space-y-2">
                        <div className="flex items-center space-x-2">
                            <Input
                                type="checkbox"
                                id="include-timestamps"
                                checked={includeTimestamps}
                                onChange={(e) => setIncludeTimestamps(e.target.checked)}
                                className="h-4 w-4"
                            />
                            <Label htmlFor="include-timestamps" className="font-normal">
                                Include Timestamps in Output
                            </Label>
                        </div>
                    </div>

                    {includeTimestamps && (
                        <div className="space-y-2">
                            <Label htmlFor="line-char-limit">Line Character Limit (for timestamped output)</Label>
                            <Input
                                id="line-char-limit"
                                type="number"
                                value={lineCharLimit}
                                onChange={(e) => setLineCharLimit(Math.max(40, Math.min(200, parseInt(e.target.value, 10) || 80)))}
                                min={40}
                                max={200}
                                disabled={!includeTimestamps}
                            />
                        </div>
                    )}

                    {isLoading && (
                        <div className="space-y-1">
                            <Label>{progress === 100 ? "Processing complete..." : "Transcribing..."}</Label>
                            <Progress value={progress} className="w-full" />
                        </div>
                    )}

                    {error && (
                        <Alert variant="destructive">
                            <AlertTitle>Transcription Error</AlertTitle>
                            <AlertDescription>{error}</AlertDescription>
                        </Alert>
                    )}

                    {transcriptionResult && (
                        <div className="space-y-2 pt-4">
                            <Label htmlFor="transcription-output">Transcription Result</Label>
                            <Textarea
                                id="transcription-output"
                                value={transcriptionResult.transcription}
                                readOnly
                                rows={10}
                                className="font-mono text-sm"
                            />
                            <p className="text-sm text-muted-foreground">File processed: {transcriptionResult.file_processed}</p>
                            <p className="text-sm text-muted-foreground">{transcriptionResult.message}</p>
                        </div>
                    )}
                </CardContent>
                <CardFooter>
                    <Button type="submit" disabled={isLoading || !file} className="w-full md:w-auto">
                        {isLoading ? "Transcribing..." : "Start Transcription"}
                    </Button>
                </CardFooter>
            </form>
        </Card>
    );
} 