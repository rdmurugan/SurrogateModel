import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Tabs,
  Tab,
  Box,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Typography,
  Slider,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  Alert,
  Switch,
  FormControlLabel,
  Divider,
  Paper
} from '@mui/material';
import {
  GridOn,
  Settings,
  PlayArrow,
  CheckCircle,
  Warning,
  Info
} from '@mui/icons-material';

interface MeshGenerationProps {
  open: boolean;
  onClose: () => void;
  onGenerate?: (meshConfig: MeshConfiguration) => void;
}

interface MeshConfiguration {
  elementType: 'tetrahedral' | 'hexahedral' | 'mixed' | 'shell' | 'beam';
  globalSize: number;
  minSize: number;
  maxSize: number;
  growthRate: number;
  curvatureAngle: number;
  enableRefinement: boolean;
  refinementRegions: RefinementRegion[];
  qualitySettings: QualitySettings;
  advancedSettings: AdvancedSettings;
}

interface RefinementRegion {
  id: string;
  type: 'sphere' | 'box' | 'cylinder' | 'surface';
  size: number;
  location: [number, number, number];
}

interface QualitySettings {
  minElementQuality: number;
  aspectRatioLimit: number;
  skewnessLimit: number;
  enableSmoothing: boolean;
  smoothingIterations: number;
}

interface AdvancedSettings {
  algorithm: 'auto' | 'delaunay' | 'frontal' | 'mmg3d';
  optimization: boolean;
  optimizationIterations: number;
  parallelization: boolean;
  memoryLimit: number;
}

const MeshGeneration: React.FC<MeshGenerationProps> = ({ open, onClose, onGenerate }) => {
  const [activeTab, setActiveTab] = useState(0);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationProgress, setGenerationProgress] = useState(0);
  const [generationStatus, setGenerationStatus] = useState<'idle' | 'running' | 'complete' | 'error'>('idle');
  const [meshStats, setMeshStats] = useState<{
    nodes: number;
    elements: number;
    minQuality: number;
    avgQuality: number;
    minSize: number;
    maxSize: number;
  } | null>(null);

  const [meshConfig, setMeshConfig] = useState<MeshConfiguration>({
    elementType: 'tetrahedral',
    globalSize: 10,
    minSize: 1,
    maxSize: 50,
    growthRate: 1.2,
    curvatureAngle: 15,
    enableRefinement: false,
    refinementRegions: [],
    qualitySettings: {
      minElementQuality: 0.3,
      aspectRatioLimit: 10,
      skewnessLimit: 0.8,
      enableSmoothing: true,
      smoothingIterations: 5
    },
    advancedSettings: {
      algorithm: 'auto',
      optimization: true,
      optimizationIterations: 10,
      parallelization: true,
      memoryLimit: 4096
    }
  });

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleGenerateMesh = async () => {
    setIsGenerating(true);
    setGenerationStatus('running');
    setGenerationProgress(0);

    // Simulate mesh generation progress
    const progressInterval = setInterval(() => {
      setGenerationProgress(prev => {
        if (prev >= 100) {
          clearInterval(progressInterval);
          setIsGenerating(false);
          setGenerationStatus('complete');
          setMeshStats({
            nodes: Math.floor(Math.random() * 50000 + 10000),
            elements: Math.floor(Math.random() * 200000 + 50000),
            minQuality: 0.2 + Math.random() * 0.3,
            avgQuality: 0.7 + Math.random() * 0.2,
            minSize: meshConfig.minSize * (0.8 + Math.random() * 0.4),
            maxSize: meshConfig.maxSize * (0.8 + Math.random() * 0.4)
          });
          return 100;
        }
        return prev + Math.random() * 10 + 2;
      });
    }, 200);

    if (onGenerate) {
      onGenerate(meshConfig);
    }
  };

  const updateMeshConfig = (updates: Partial<MeshConfiguration>) => {
    setMeshConfig(prev => ({ ...prev, ...updates }));
  };

  const getQualityColor = (quality: number) => {
    if (quality < 0.3) return 'error';
    if (quality < 0.6) return 'warning';
    return 'success';
  };

  const getElementTypeDescription = (type: string) => {
    switch (type) {
      case 'tetrahedral': return 'General purpose 3D elements, good for complex geometries';
      case 'hexahedral': return 'High-quality structured elements, best accuracy';
      case 'mixed': return 'Combination of element types for optimal performance';
      case 'shell': return 'Thin-walled structures and surfaces';
      case 'beam': return 'Frame and truss structures';
      default: return '';
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        <Typography variant="h6" component="div" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <GridOn />
          Mesh Generation
        </Typography>
      </DialogTitle>
      <DialogContent>
        <Box sx={{ width: '100%' }}>
          <Tabs value={activeTab} onChange={handleTabChange}>
            <Tab label="Element Settings" />
            <Tab label="Size Controls" />
            <Tab label="Quality Settings" />
            <Tab label="Advanced" />
            <Tab label="Generate" />
          </Tabs>

          {activeTab === 0 && (
            <Box sx={{ mt: 3 }}>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <FormControl fullWidth margin="normal">
                    <InputLabel>Element Type</InputLabel>
                    <Select
                      value={meshConfig.elementType}
                      label="Element Type"
                      onChange={(e) => updateMeshConfig({ elementType: e.target.value as any })}
                    >
                      <MenuItem value="tetrahedral">Tetrahedral (3D)</MenuItem>
                      <MenuItem value="hexahedral">Hexahedral (3D)</MenuItem>
                      <MenuItem value="mixed">Mixed Elements</MenuItem>
                      <MenuItem value="shell">Shell Elements</MenuItem>
                      <MenuItem value="beam">Beam Elements</MenuItem>
                    </Select>
                  </FormControl>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    {getElementTypeDescription(meshConfig.elementType)}
                  </Typography>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 2, mt: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>Element Type Features</Typography>
                    <Box sx={{ '& > div': { display: 'flex', justifyContent: 'space-between', py: 0.5 } }}>
                      <div><span>Geometry Support:</span> <strong>Complex 3D</strong></div>
                      <div><span>Accuracy:</span> <strong>High</strong></div>
                      <div><span>Generation Speed:</span> <strong>Fast</strong></div>
                      <div><span>Memory Usage:</span> <strong>Moderate</strong></div>
                    </Box>
                  </Paper>
                </Grid>
              </Grid>
            </Box>
          )}

          {activeTab === 1 && (
            <Box sx={{ mt: 3 }}>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>Global Size Controls</Typography>
                  <Box sx={{ mb: 3 }}>
                    <Typography gutterBottom>Global Element Size: {meshConfig.globalSize} mm</Typography>
                    <Slider
                      value={meshConfig.globalSize}
                      onChange={(_, value) => updateMeshConfig({ globalSize: value as number })}
                      min={1}
                      max={100}
                      valueLabelDisplay="auto"
                      marks={[
                        { value: 1, label: 'Fine' },
                        { value: 50, label: 'Medium' },
                        { value: 100, label: 'Coarse' }
                      ]}
                    />
                  </Box>
                  <TextField
                    fullWidth
                    label="Minimum Size (mm)"
                    type="number"
                    value={meshConfig.minSize}
                    onChange={(e) => updateMeshConfig({ minSize: Number(e.target.value) })}
                    margin="normal"
                  />
                  <TextField
                    fullWidth
                    label="Maximum Size (mm)"
                    type="number"
                    value={meshConfig.maxSize}
                    onChange={(e) => updateMeshConfig({ maxSize: Number(e.target.value) })}
                    margin="normal"
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>Refinement Controls</Typography>
                  <Box sx={{ mb: 3 }}>
                    <Typography gutterBottom>Growth Rate: {meshConfig.growthRate}</Typography>
                    <Slider
                      value={meshConfig.growthRate}
                      onChange={(_, value) => updateMeshConfig({ growthRate: value as number })}
                      min={1.1}
                      max={2.0}
                      step={0.1}
                      valueLabelDisplay="auto"
                      marks={[
                        { value: 1.1, label: 'Smooth' },
                        { value: 1.5, label: 'Balanced' },
                        { value: 2.0, label: 'Aggressive' }
                      ]}
                    />
                  </Box>
                  <TextField
                    fullWidth
                    label="Curvature Angle (degrees)"
                    type="number"
                    value={meshConfig.curvatureAngle}
                    onChange={(e) => updateMeshConfig({ curvatureAngle: Number(e.target.value) })}
                    margin="normal"
                    helperText="Smaller angles create finer mesh on curved surfaces"
                  />
                  <FormControlLabel
                    control={
                      <Switch
                        checked={meshConfig.enableRefinement}
                        onChange={(e) => updateMeshConfig({ enableRefinement: e.target.checked })}
                      />
                    }
                    label="Enable Local Refinement"
                  />
                </Grid>
              </Grid>
            </Box>
          )}

          {activeTab === 2 && (
            <Box sx={{ mt: 3 }}>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>Quality Thresholds</Typography>
                  <Box sx={{ mb: 3 }}>
                    <Typography gutterBottom>
                      Minimum Element Quality: {meshConfig.qualitySettings.minElementQuality.toFixed(2)}
                    </Typography>
                    <Slider
                      value={meshConfig.qualitySettings.minElementQuality}
                      onChange={(_, value) => updateMeshConfig({
                        qualitySettings: { ...meshConfig.qualitySettings, minElementQuality: value as number }
                      })}
                      min={0.1}
                      max={0.9}
                      step={0.05}
                      valueLabelDisplay="auto"
                      marks={[
                        { value: 0.1, label: 'Low' },
                        { value: 0.3, label: 'Good' },
                        { value: 0.6, label: 'High' },
                        { value: 0.9, label: 'Excellent' }
                      ]}
                    />
                  </Box>
                  <TextField
                    fullWidth
                    label="Aspect Ratio Limit"
                    type="number"
                    value={meshConfig.qualitySettings.aspectRatioLimit}
                    onChange={(e) => updateMeshConfig({
                      qualitySettings: { ...meshConfig.qualitySettings, aspectRatioLimit: Number(e.target.value) }
                    })}
                    margin="normal"
                  />
                  <TextField
                    fullWidth
                    label="Skewness Limit"
                    type="number"
                    inputProps={{ step: 0.1, min: 0, max: 1 }}
                    value={meshConfig.qualitySettings.skewnessLimit}
                    onChange={(e) => updateMeshConfig({
                      qualitySettings: { ...meshConfig.qualitySettings, skewnessLimit: Number(e.target.value) }
                    })}
                    margin="normal"
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>Mesh Improvement</Typography>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={meshConfig.qualitySettings.enableSmoothing}
                        onChange={(e) => updateMeshConfig({
                          qualitySettings: { ...meshConfig.qualitySettings, enableSmoothing: e.target.checked }
                        })}
                      />
                    }
                    label="Enable Smoothing"
                  />
                  {meshConfig.qualitySettings.enableSmoothing && (
                    <TextField
                      fullWidth
                      label="Smoothing Iterations"
                      type="number"
                      value={meshConfig.qualitySettings.smoothingIterations}
                      onChange={(e) => updateMeshConfig({
                        qualitySettings: { ...meshConfig.qualitySettings, smoothingIterations: Number(e.target.value) }
                      })}
                      margin="normal"
                    />
                  )}
                  <Alert severity="info" sx={{ mt: 2 }}>
                    Higher quality settings may increase generation time but improve analysis accuracy.
                  </Alert>
                </Grid>
              </Grid>
            </Box>
          )}

          {activeTab === 3 && (
            <Box sx={{ mt: 3 }}>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>Algorithm Settings</Typography>
                  <FormControl fullWidth margin="normal">
                    <InputLabel>Meshing Algorithm</InputLabel>
                    <Select
                      value={meshConfig.advancedSettings.algorithm}
                      label="Meshing Algorithm"
                      onChange={(e) => updateMeshConfig({
                        advancedSettings: { ...meshConfig.advancedSettings, algorithm: e.target.value as any }
                      })}
                    >
                      <MenuItem value="auto">Automatic Selection</MenuItem>
                      <MenuItem value="delaunay">Delaunay Triangulation</MenuItem>
                      <MenuItem value="frontal">Frontal Algorithm</MenuItem>
                      <MenuItem value="mmg3d">MMG3D Remesher</MenuItem>
                    </Select>
                  </FormControl>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={meshConfig.advancedSettings.optimization}
                        onChange={(e) => updateMeshConfig({
                          advancedSettings: { ...meshConfig.advancedSettings, optimization: e.target.checked }
                        })}
                      />
                    }
                    label="Enable Optimization"
                  />
                  {meshConfig.advancedSettings.optimization && (
                    <TextField
                      fullWidth
                      label="Optimization Iterations"
                      type="number"
                      value={meshConfig.advancedSettings.optimizationIterations}
                      onChange={(e) => updateMeshConfig({
                        advancedSettings: { ...meshConfig.advancedSettings, optimizationIterations: Number(e.target.value) }
                      })}
                      margin="normal"
                    />
                  )}
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>Performance Settings</Typography>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={meshConfig.advancedSettings.parallelization}
                        onChange={(e) => updateMeshConfig({
                          advancedSettings: { ...meshConfig.advancedSettings, parallelization: e.target.checked }
                        })}
                      />
                    }
                    label="Enable Parallel Processing"
                  />
                  <TextField
                    fullWidth
                    label="Memory Limit (MB)"
                    type="number"
                    value={meshConfig.advancedSettings.memoryLimit}
                    onChange={(e) => updateMeshConfig({
                      advancedSettings: { ...meshConfig.advancedSettings, memoryLimit: Number(e.target.value) }
                    })}
                    margin="normal"
                  />
                </Grid>
              </Grid>
            </Box>
          )}

          {activeTab === 4 && (
            <Box sx={{ mt: 3 }}>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        <Settings sx={{ mr: 1, verticalAlign: 'middle' }} />
                        Mesh Configuration Summary
                      </Typography>
                      <Divider sx={{ my: 2 }} />
                      <Box sx={{ '& > div': { display: 'flex', justifyContent: 'space-between', py: 0.5 } }}>
                        <div><span>Element Type:</span> <strong>{meshConfig.elementType}</strong></div>
                        <div><span>Global Size:</span> <strong>{meshConfig.globalSize} mm</strong></div>
                        <div><span>Size Range:</span> <strong>{meshConfig.minSize} - {meshConfig.maxSize} mm</strong></div>
                        <div><span>Growth Rate:</span> <strong>{meshConfig.growthRate}</strong></div>
                        <div><span>Min Quality:</span> <strong>{meshConfig.qualitySettings.minElementQuality}</strong></div>
                        <div><span>Algorithm:</span> <strong>{meshConfig.advancedSettings.algorithm}</strong></div>
                      </Box>
                    </CardContent>
                  </Card>

                  <Box sx={{ mt: 3, textAlign: 'center' }}>
                    <Button
                      variant="contained"
                      size="large"
                      startIcon={isGenerating ? <Settings className="spin" /> : <PlayArrow />}
                      onClick={handleGenerateMesh}
                      disabled={isGenerating}
                      sx={{ minWidth: 200 }}
                    >
                      {isGenerating ? 'Generating...' : 'Generate Mesh'}
                    </Button>
                  </Box>
                </Grid>

                <Grid item xs={12} md={6}>
                  {generationStatus !== 'idle' && (
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Generation Status
                        </Typography>
                        {generationStatus === 'running' && (
                          <Box>
                            <LinearProgress variant="determinate" value={generationProgress} sx={{ mb: 2 }} />
                            <Typography variant="body2" color="text.secondary">
                              Progress: {Math.round(generationProgress)}%
                            </Typography>
                          </Box>
                        )}
                        {generationStatus === 'complete' && meshStats && (
                          <Box>
                            <Alert severity="success" sx={{ mb: 2 }}>
                              <CheckCircle sx={{ mr: 1 }} />
                              Mesh generation completed successfully!
                            </Alert>
                            <Typography variant="subtitle2" gutterBottom>Mesh Statistics</Typography>
                            <Box sx={{ '& > div': { display: 'flex', justifyContent: 'space-between', py: 0.5 } }}>
                              <div><span>Nodes:</span> <strong>{meshStats.nodes.toLocaleString()}</strong></div>
                              <div><span>Elements:</span> <strong>{meshStats.elements.toLocaleString()}</strong></div>
                              <div>
                                <span>Min Quality:</span>
                                <strong style={{ color: getQualityColor(meshStats.minQuality) === 'error' ? '#d32f2f' : getQualityColor(meshStats.minQuality) === 'warning' ? '#ed6c02' : '#2e7d32' }}>
                                  {meshStats.minQuality.toFixed(3)}
                                </strong>
                              </div>
                              <div><span>Avg Quality:</span> <strong>{meshStats.avgQuality.toFixed(3)}</strong></div>
                              <div><span>Size Range:</span> <strong>{meshStats.minSize.toFixed(1)} - {meshStats.maxSize.toFixed(1)} mm</strong></div>
                            </Box>
                          </Box>
                        )}
                      </CardContent>
                    </Card>
                  )}
                </Grid>
              </Grid>
            </Box>
          )}
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
        {generationStatus === 'complete' && (
          <Button variant="contained" onClick={onClose}>
            Accept Mesh
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default MeshGeneration;