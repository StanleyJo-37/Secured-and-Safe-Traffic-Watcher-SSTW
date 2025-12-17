"use client";

import { useState, useRef, useCallback } from "react";
import { Upload, Video, AlertTriangle, CheckCircle, Clock, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

interface AnalysisResult {
  safety_score: number;
  traffic_density: string;
  violations_detected: number;
  details: Array<{
    timestamp: string;
    type: string;
    severity: string;
  }>;
}

const API_BASE = "http://localhost:5000";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFileSelect = useCallback((selectedFile: File) => {
    if (selectedFile && selectedFile.type.startsWith("video/")) {
      setFile(selectedFile);
      setVideoPreview(URL.createObjectURL(selectedFile));
      setError(null);
    } else {
      setError("Please select a valid video file");
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile) {
        handleFileSelect(droppedFile);
      }
    },
    [handleFileSelect]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selectedFile = e.target.files?.[0];
      if (selectedFile) {
        handleFileSelect(selectedFile);
      }
    },
    [handleFileSelect]
  );

  const handleUpload = async () => {
    if (!file) return;

    setIsUploading(true);
    setUploadProgress(0);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      // Simulate progress for better UX
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 500);

      const response = await fetch(`${API_BASE}/process_video`, {
        method: "POST",
        body: formData,
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Upload failed");
      }

      // Fetch analysis results
      const analysisResponse = await fetch(`${API_BASE}/analysis_result`);
      if (analysisResponse.ok) {
        const results = await analysisResponse.json();
        setAnalysisResult(results);
        setShowResults(true);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsUploading(false);
    }
  };

  const resetUpload = () => {
    setFile(null);
    setVideoPreview(null);
    setUploadProgress(0);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case "high":
        return "text-error bg-error/10";
      case "medium":
        return "text-warning bg-warning/10";
      case "low":
        return "text-success bg-success/10";
      default:
        return "text-secondary-text bg-secondary-text/10";
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return "text-success";
    if (score >= 60) return "text-warning";
    return "text-error";
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-secondary-text/10 bg-surface">
        <div className="mx-auto max-w-5xl px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent">
              <Video className="h-5 w-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-primary-text">SSTW</h1>
              <p className="text-xs text-secondary-text">Traffic Watcher</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-5xl px-6 py-12">
        <div className="mb-8">
          <h2 className="text-2xl font-semibold text-primary-text">
            Traffic Analysis
          </h2>
          <p className="mt-2 text-secondary-text">
            Upload a video to analyze traffic patterns and detect violations
          </p>
        </div>

        {/* Upload Card */}
        <Card className="overflow-hidden">
          <CardHeader>
            <CardTitle className="text-base">Upload Video</CardTitle>
            <CardDescription>
              Drag and drop a video file or click to browse
            </CardDescription>
          </CardHeader>
          <CardContent>
            {!file ? (
              <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => fileInputRef.current?.click()}
                className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-12 transition-all duration-200 ${
                  isDragOver
                    ? "border-accent bg-accent/5"
                    : "border-secondary-text/20 hover:border-accent/50 hover:bg-accent/5"
                }`}
              >
                <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-accent/10">
                  <Upload className="h-6 w-6 text-accent" />
                </div>
                <p className="mb-1 text-sm font-medium text-primary-text">
                  Drop your video here
                </p>
                <p className="text-xs text-secondary-text">
                  Supports MP4, MOV, AVI up to 512MB
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/*"
                  onChange={handleInputChange}
                  className="hidden"
                />
              </div>
            ) : (
              <div className="space-y-4">
                {/* Video Preview */}
                <div className="relative overflow-hidden rounded-xl bg-primary-text/5">
                  <video
                    src={videoPreview!}
                    controls
                    className="mx-auto max-h-[400px] w-full object-contain"
                  />
                  <button
                    onClick={resetUpload}
                    className="absolute right-3 top-3 flex h-8 w-8 items-center justify-center rounded-full bg-primary-text/80 text-white transition-colors hover:bg-primary-text"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>

                {/* File Info */}
                <div className="flex items-center justify-between rounded-lg bg-secondary-text/5 px-4 py-3">
                  <div className="flex items-center gap-3">
                    <Video className="h-5 w-5 text-accent" />
                    <div>
                      <p className="text-sm font-medium text-primary-text">
                        {file.name}
                      </p>
                      <p className="text-xs text-secondary-text">
                        {(file.size / (1024 * 1024)).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                </div>

                {/* Progress Bar */}
                {isUploading && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-secondary-text">Processing...</span>
                      <span className="font-medium text-accent">
                        {uploadProgress}%
                      </span>
                    </div>
                    <Progress value={uploadProgress} />
                  </div>
                )}

                {/* Error Message */}
                {error && (
                  <div className="flex items-center gap-2 rounded-lg bg-error/10 px-4 py-3 text-sm text-error">
                    <AlertTriangle className="h-4 w-4" />
                    {error}
                  </div>
                )}

                {/* Action Buttons */}
                <div className="flex gap-3">
                  <Button
                    onClick={handleUpload}
                    disabled={isUploading}
                    className="flex-1"
                  >
                    {isUploading ? (
                      <>
                        <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Upload className="h-4 w-4" />
                        Analyze Video
                      </>
                    )}
                  </Button>
                  <Button variant="outline" onClick={resetUpload} disabled={isUploading}>
                    Cancel
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </main>

      {/* Results Dialog */}
      <Dialog open={showResults} onOpenChange={setShowResults}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-success" />
              Analysis Complete
            </DialogTitle>
            <DialogDescription>
              Traffic analysis results for your uploaded video
            </DialogDescription>
          </DialogHeader>

          {analysisResult && (
            <div className="space-y-6">
              {/* Score Grid */}
              <div className="grid grid-cols-3 gap-4">
                <div className="rounded-xl bg-secondary-text/5 p-4 text-center">
                  <p className="text-xs text-secondary-text">Safety Score</p>
                  <p
                    className={`mt-1 text-3xl font-bold ${getScoreColor(
                      analysisResult.safety_score
                    )}`}
                  >
                    {analysisResult.safety_score}
                  </p>
                </div>
                <div className="rounded-xl bg-secondary-text/5 p-4 text-center">
                  <p className="text-xs text-secondary-text">Traffic Density</p>
                  <p className="mt-1 text-lg font-semibold text-primary-text">
                    {analysisResult.traffic_density}
                  </p>
                </div>
                <div className="rounded-xl bg-secondary-text/5 p-4 text-center">
                  <p className="text-xs text-secondary-text">Violations</p>
                  <p className="mt-1 text-3xl font-bold text-error">
                    {analysisResult.violations_detected}
                  </p>
                </div>
              </div>

              {/* Violations List */}
              {analysisResult.details.length > 0 && (
                <div>
                  <h4 className="mb-3 text-sm font-medium text-primary-text">
                    Detected Violations
                  </h4>
                  <div className="space-y-2">
                    {analysisResult.details.map((detail, index) => (
                      <div
                        key={index}
                        className="flex items-center justify-between rounded-lg border border-secondary-text/10 px-4 py-3"
                      >
                        <div className="flex items-center gap-3">
                          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-secondary-text/10">
                            <Clock className="h-4 w-4 text-secondary-text" />
                          </div>
                          <div>
                            <p className="text-sm font-medium text-primary-text">
                              {detail.type}
                            </p>
                            <p className="text-xs text-secondary-text">
                              {detail.timestamp}
                            </p>
                          </div>
                        </div>
                        <span
                          className={`rounded-full px-2 py-0.5 text-xs font-medium ${getSeverityColor(
                            detail.severity
                          )}`}
                        >
                          {detail.severity}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Close Button */}
              <Button
                onClick={() => setShowResults(false)}
                className="w-full"
                variant="outline"
              >
                Close
              </Button>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
