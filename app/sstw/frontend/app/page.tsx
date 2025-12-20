"use client";

import { useState, useRef, useCallback } from "react";
import { Upload, Video, Image, AlertTriangle, CheckCircle, Clock, X } from "lucide-react";
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
  car_count?: number;
  details: Array<{
    timestamp: string;
    type: string;
    severity: string;
  }>;
}

const API_BASE = "http://localhost:5000";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [filePreview, setFilePreview] = useState<string | null>(null);
  const [fileType, setFileType] = useState<"video" | "image" | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFileSelect = useCallback((selectedFile: File) => {
    const isVideo = selectedFile.type.startsWith("video/");
    const isImage = selectedFile.type.startsWith("image/");
    
    if (selectedFile && (isVideo || isImage)) {
      setFile(selectedFile);
      setFilePreview(URL.createObjectURL(selectedFile));
      setFileType(isVideo ? "video" : "image");
      setError(null);
      setResultImage(null);
    } else {
      setError("Please select a valid video or image file");
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
    if (!file || !fileType) return;

    setIsUploading(true);
    setUploadProgress(0);
    setError(null);
    setResultImage(null);

    const formData = new FormData();
    formData.append("file", file);

    const endpoint = fileType === "video" ? "/process_video" : "/process_image";

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

      const response = await fetch(`${API_BASE}${endpoint}`, {
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

      // For images, also fetch the result image
      if (fileType === "image") {
        setResultImage(`${API_BASE}/result_image?t=${Date.now()}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsUploading(false);
    }
  };

  const resetUpload = () => {
    setFile(null);
    setFilePreview(null);
    setFileType(null);
    setUploadProgress(0);
    setError(null);
    setResultImage(null);
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

  const getDensityColor = (density: string) => {
    switch (density.toLowerCase()) {
      case "high":
        return "bg-error/20 text-error border-error/30";
      case "medium":
        return "bg-warning/20 text-warning border-warning/30";
      case "low":
        return "bg-success/20 text-success border-success/30";
      default:
        return "bg-secondary-text/20 text-secondary-text border-secondary-text/30";
    }
  };

  const getSeverityLevel = (count: number) => {
    if (count === 0) return { level: "Clear", description: "No vehicles detected" };
    if (count <= 3) return { level: "Light", description: "Light traffic flow" };
    if (count <= 7) return { level: "Moderate", description: "Moderate traffic" };
    if (count <= 15) return { level: "Heavy", description: "Heavy traffic detected" };
    return { level: "Congested", description: "Traffic congestion" };
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
            Upload a video or image to analyze traffic patterns and detect violations
          </p>
        </div>

        {/* Upload Card */}
        <Card className="overflow-hidden">
          <CardHeader>
            <CardTitle className="text-base">Upload Video or Image</CardTitle>
            <CardDescription>
              Drag and drop a video or image file, or click to browse
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
                  Drop your file here
                </p>
                <p className="text-xs text-secondary-text">
                  Supports MP4, MOV, AVI, JPG, PNG up to 512MB
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/*,image/*"
                  onChange={handleInputChange}
                  className="hidden"
                />
              </div>
            ) : (
              <div className="space-y-4">
                {/* File Preview */}
                <div className="relative overflow-hidden rounded-xl bg-primary-text/5">
                  {fileType === "video" ? (
                    <video
                      src={filePreview!}
                      controls
                      className="mx-auto max-h-[400px] w-full object-contain"
                    />
                  ) : (
                    <img
                      src={filePreview!}
                      alt="Preview"
                      className="mx-auto max-h-[400px] w-full object-contain"
                    />
                  )}
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
                    {fileType === "video" ? (
                      <Video className="h-5 w-5 text-accent" />
                    ) : (
                      <Image className="h-5 w-5 text-accent" />
                    )}
                    <div>
                      <p className="text-sm font-medium text-primary-text">
                        {file.name}
                      </p>
                      <p className="text-xs text-secondary-text">
                        {(file.size / (1024 * 1024)).toFixed(2)} MB · {fileType === "video" ? "Video" : "Image"}
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
                        Analyze {fileType === "video" ? "Video" : "Image"}
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
              Traffic analysis results for your uploaded {fileType === "video" ? "video" : "image"}
            </DialogDescription>
          </DialogHeader>

          {analysisResult && (
            <div className="space-y-6">
              {/* Result Image with Detections */}
              {resultImage && (
                <div className="overflow-hidden rounded-xl border border-secondary-text/10">
                  <img
                    src={resultImage}
                    alt="Processed result with detections"
                    className="w-full object-contain"
                  />
                </div>
              )}

              {/* Main Stats */}
              <div className="grid grid-cols-2 gap-4">
                {/* Vehicle Count */}
                <div className="rounded-xl bg-accent/10 border border-accent/20 p-5 text-center">
                  <p className="text-xs text-secondary-text uppercase tracking-wide">Vehicles Detected</p>
                  <p className="mt-2 text-5xl font-bold text-accent">
                    {analysisResult.violations_detected}
                  </p>
                  <p className="mt-1 text-sm text-secondary-text">
                    {getSeverityLevel(analysisResult.violations_detected).description}
                  </p>
                </div>

                {/* Traffic Severity */}
                <div className="rounded-xl bg-secondary-text/5 border border-secondary-text/10 p-5 text-center">
                  <p className="text-xs text-secondary-text uppercase tracking-wide">Traffic Severity</p>
                  <p className={`mt-2 text-3xl font-bold ${getScoreColor(analysisResult.safety_score)}`}>
                    {getSeverityLevel(analysisResult.violations_detected).level}
                  </p>
                  <div className={`mt-2 inline-block rounded-full px-3 py-1 text-xs font-medium border ${getDensityColor(analysisResult.traffic_density)}`}>
                    {analysisResult.traffic_density} Density
                  </div>
                </div>
              </div>

              {/* Safety Score Bar */}
              <div className="rounded-xl bg-secondary-text/5 p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-secondary-text">Road Safety Score</span>
                  <span className={`text-lg font-bold ${getScoreColor(analysisResult.safety_score)}`}>
                    {analysisResult.safety_score}/100
                  </span>
                </div>
                <div className="h-2 w-full rounded-full bg-secondary-text/10 overflow-hidden">
                  <div 
                    className={`h-full rounded-full transition-all duration-500 ${
                      analysisResult.safety_score >= 80 ? 'bg-success' : 
                      analysisResult.safety_score >= 60 ? 'bg-warning' : 'bg-error'
                    }`}
                    style={{ width: `${analysisResult.safety_score}%` }}
                  />
                </div>
                <p className="mt-2 text-xs text-secondary-text">
                  {analysisResult.safety_score >= 80 
                    ? "Low risk - Safe traffic conditions" 
                    : analysisResult.safety_score >= 60 
                    ? "Moderate risk - Exercise caution" 
                    : "High risk - Heavy traffic detected"}
                </p>
              </div>

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
