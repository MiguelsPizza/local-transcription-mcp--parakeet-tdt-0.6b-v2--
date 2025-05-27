import { Card, CardContent, CardHeader, CardTitle } from "@workspace/ui/components/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@workspace/ui/components/tabs";
import { HardwareInfo } from "./components/HardwareInfo";
import { ModelInfo } from "./components/ModelInfo";
import { TranscriptionForm } from "./components/TranscriptionForm";

function App() {
  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col items-center p-4 md:p-8">
      <header className="mb-8">
        <h1 className="text-4xl font-bold tracking-tight text-center">
          Parakeet Transcription Dashboard
        </h1>
        <p className="text-center text-muted-foreground mt-2">
          Transcribe audio, view model details, and check system hardware.
        </p>
      </header>

      <Tabs defaultValue="transcribe" className="w-full max-w-4xl">
        <TabsList className="grid w-full grid-cols-3 mb-6">
          <TabsTrigger value="transcribe">Transcribe Audio</TabsTrigger>
          <TabsTrigger value="model-info">ASR Model Info</TabsTrigger>
          <TabsTrigger value="hardware-info">System Hardware</TabsTrigger>
        </TabsList>

        <TabsContent value="transcribe">
          <TranscriptionForm />
        </TabsContent>

        <TabsContent value="model-info">
          <ModelInfo />
        </TabsContent>

        <TabsContent value="hardware-info">
          <HardwareInfo />
        </TabsContent>
      </Tabs>
      
      <footer className="mt-12 text-center text-sm text-muted-foreground">
        <p>&copy; {new Date().getFullYear()} Transcription Service. Powered by NVIDIA Parakeet & FastMCP.</p>
        <p>Frontend by shadcn/ui, Vite, and React.</p>
      </footer>
    </div>
  );
}

export default App;
