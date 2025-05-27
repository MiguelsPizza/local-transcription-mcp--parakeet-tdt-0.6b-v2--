import { Alert, AlertDescription, AlertTitle } from "@workspace/ui/components/alert";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@workspace/ui/components/card";
import { Skeleton } from "@workspace/ui/components/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@workspace/ui/components/table";
import { useEffect, useState } from "react";
import { getHardwareInfo, type SystemHardwareResponse } from "../lib/api";

export function HardwareInfo() {
  const [hardwareInfo, setHardwareInfo] = useState<SystemHardwareResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHardwareInfo = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const data = await getHardwareInfo();
        setHardwareInfo(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "An unknown error occurred.");
      } finally {
        setIsLoading(false);
      }
    };
    fetchHardwareInfo();
  }, []);

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-1/2" />
          <Skeleton className="h-4 w-3/4 mt-1" />
        </CardHeader>
        <CardContent className="space-y-4">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-20 w-full" />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Error Fetching Hardware Info</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  if (!hardwareInfo) {
    return (
      <Alert>
        <AlertTitle>No Hardware Information</AlertTitle>
        <AlertDescription>No hardware information could be retrieved.</AlertDescription>
      </Alert>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>System Hardware Specifications</CardTitle>
        <CardDescription>Detailed information about the server's hardware.</CardDescription>
        {hardwareInfo.error && (
          <Alert variant="default" className="mt-2">
            <AlertTitle>Partial Data Warning</AlertTitle>
            <AlertDescription>{hardwareInfo.error}</AlertDescription>
          </Alert>
        )}
        {hardwareInfo.error_partial_results && (
          <Alert variant="default" className="mt-2">
            <AlertTitle>Partial Data Warning</AlertTitle>
            <AlertDescription>{hardwareInfo.error_partial_results}</AlertDescription>
          </Alert>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <InfoItem label="OS Platform" value={hardwareInfo.os_platform} />
          <InfoItem label="OS Version" value={hardwareInfo.os_version} />
          <InfoItem label="OS Release" value={hardwareInfo.os_release} />
          <InfoItem label="Architecture" value={hardwareInfo.architecture} />
          <InfoItem label="CPU Model" value={hardwareInfo.cpu_model} />
          <InfoItem label="CPU Physical Cores" value={hardwareInfo.cpu_physical_cores} />
          <InfoItem label="CPU Logical Cores" value={hardwareInfo.cpu_logical_cores} />
          <InfoItem label="CPU Max Frequency (GHz)" value={hardwareInfo.cpu_frequency_max_ghz} />
          <InfoItem label="Total RAM (GB)" value={hardwareInfo.ram_total_gb} />
          <InfoItem label="Available RAM (GB)" value={hardwareInfo.ram_available_gb} />
          <InfoItem label="CUDA Available" value={hardwareInfo.cuda_available === null ? "N/A" : hardwareInfo.cuda_available ? "Yes" : "No"} />
          {hardwareInfo.cuda_available && (
            <InfoItem label="CUDA Version" value={hardwareInfo.cuda_version} />
          )}
          <InfoItem label="GPU Count" value={hardwareInfo.gpu_count} />
        </div>

        {hardwareInfo.gpus && hardwareInfo.gpus.length > 0 && (
          <div>
            <h3 className="text-lg font-semibold mt-4 mb-2">GPU Details</h3>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Memory (GB)</TableHead>
                  <TableHead>CUDA Capability</TableHead>
                  <TableHead>Notes</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {hardwareInfo.gpus.map((gpu, index) => (
                  <TableRow key={index}>
                    <TableCell>{gpu.name}</TableCell>
                    <TableCell>{gpu.memory_total_gb}</TableCell>
                    <TableCell>{gpu.cuda_capability ? `${gpu.cuda_capability[0]}.${gpu.cuda_capability[1]}` : "N/A"}</TableCell>
                    <TableCell>{gpu.notes || "-"}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
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