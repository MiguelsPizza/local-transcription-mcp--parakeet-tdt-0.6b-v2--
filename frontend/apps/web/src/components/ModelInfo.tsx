import { Alert, AlertDescription, AlertTitle } from "@workspace/ui/components/alert";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@workspace/ui/components/card";
import { Skeleton } from "@workspace/ui/components/skeleton";
import { useEffect, useState } from "react";
import { getModelInfo, type ModelInfoResponse } from "../lib/api";

export function ModelInfo() {
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchModelInfo = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const data = await getModelInfo();
        setModelInfo(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "An unknown error occurred.");
      } finally {
        setIsLoading(false);
      }
    };
    fetchModelInfo();
  }, []);

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-1/2" />
          <Skeleton className="h-4 w-3/4 mt-1" />
        </CardHeader>
        <CardContent className="space-y-2">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-5/6" />
          <Skeleton className="h-4 w-full" />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Error Fetching Model Info</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  if (!modelInfo) {
    return (
      <Alert>
        <AlertTitle>No Model Information</AlertTitle>
        <AlertDescription>No ASR model information could be retrieved.</AlertDescription>
      </Alert>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>ASR Model Information</CardTitle>
        <CardDescription>Details about the Automatic Speech Recognition model being used.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <InfoItem label="Model Name" value={modelInfo.model_name} />
        <InfoItem label="Status" value={modelInfo.status} />
        <InfoItem label="Input Requirements" value={modelInfo.input_requirements} />
        <InfoItem label="Output Type" value={modelInfo.output_type} />
        <InfoItem label="License" value={modelInfo.license} />
        <InfoItem label="Note" value={modelInfo.note} />
      </CardContent>
    </Card>
  );
}

interface InfoItemProps {
  label: string;
  value: string | number | boolean | null | undefined;
}

function InfoItem({ label, value }: InfoItemProps) {
  return (
    <div className="flex flex-col p-2 border rounded-md bg-muted/30">
      <span className="text-sm font-medium text-muted-foreground">{label}</span>
      <span className="text-md font-semibold">{value === null || value === undefined ? "N/A" : String(value)}</span>
    </div>
  );
} 