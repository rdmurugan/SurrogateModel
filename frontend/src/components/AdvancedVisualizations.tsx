import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  Tabs,
  Tab,
  useTheme,
  alpha,
  IconButton,
  Tooltip,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Button,
  Chip,
} from '@mui/material';
import {
  Settings,
  Visibility,
  Download,
  PlayArrow,
  Pause,
  Stop,
  Refresh,
  Analytics,
  ScatterPlot,
  ShowChart,
  BubbleChart,
  Memory,
  Timeline,
  TrendingUp,
} from '@mui/icons-material';
import Plot from 'react-plotly.js';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`visualization-tabpanel-${index}`}
      aria-labelledby={`visualization-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export default function AdvancedVisualizations() {
  const theme = useTheme();
  const [selectedTab, setSelectedTab] = useState(0);
  const [realTimeMode, setRealTimeMode] = useState(true);
  const [selectedModel, setSelectedModel] = useState('bayesian');
  const [uncertaintyThreshold, setUncertaintyThreshold] = useState(0.1);
  const [meshResolution, setMeshResolution] = useState(50);
  const [showControls, setShowControls] = useState(true);
  const [stressScale, setStressScale] = useState(1);
  const [contourLevels, setContourLevels] = useState(10);
  const [animationSpeed, setAnimationSpeed] = useState(1);
  const [showColorbar, setShowColorbar] = useState(true);
  const [meshOpacity, setMeshOpacity] = useState(0.7);
  const [isAnimating, setIsAnimating] = useState(false);

  // Interactive control handlers
  const handleToggleControls = () => {
    setShowControls(!showControls);
    console.log('Controls visibility:', !showControls);
  };

  const handleStartAnimation = () => {
    setIsAnimating(true);
    console.log('Starting animation...');
    // TODO: Implement animation logic
    setTimeout(() => setIsAnimating(false), 3000); // Demo animation for 3 seconds
  };

  const handleStopAnimation = () => {
    setIsAnimating(false);
    console.log('Stopping animation...');
  };

  const handleExportVisualization = () => {
    console.log('Exporting visualization...');
    // TODO: Implement export functionality
  };

  const handleResetView = () => {
    setStressScale(1);
    setContourLevels(10);
    setMeshOpacity(0.7);
    setShowColorbar(true);
    console.log('View reset to defaults');
  };

  // Advanced 3D Mesh Visualization Data
  const generate3DMeshData = () => {
    const x = [];
    const y = [];
    const z = [];
    const colors = [];

    for (let i = 0; i < meshResolution; i++) {
      for (let j = 0; j < meshResolution; j++) {
        x.push(i / 10);
        y.push(j / 10);
        z.push(Math.sin(i / 5) * Math.cos(j / 5) + Math.random() * 0.1);
        colors.push(Math.sqrt(Math.pow(i - meshResolution/2, 2) + Math.pow(j - meshResolution/2, 2)));
      }
    }

    return {
      data: [{
        x: x,
        y: y,
        z: z,
        mode: 'markers' as const,
        marker: {
          size: 3,
          color: colors,
          colorscale: 'Viridis',
          showscale: true,
          colorbar: { title: { text: 'Stress (MPa)' } }
        },
        type: 'scatter3d' as const,
        name: 'Mesh Nodes'
      }],
      layout: {
        title: { text: '3D Mesh Stress Analysis', font: { size: 16, family: 'Roboto' } },
        scene: {
          xaxis: { title: 'X Position (m)' },
          yaxis: { title: 'Y Position (m)' },
          zaxis: { title: 'Z Position (m)' },
          camera: {
            eye: { x: 1.5, y: 1.5, z: 1.5 }
          },
          bgcolor: 'rgba(0,0,0,0)'
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 50, r: 20, b: 50, l: 50 }
      },
      config: { responsive: true, displayModeBar: true }
    };
  };

  // Uncertainty Quantification Heatmap
  const generateUncertaintyHeatmap = () => {
    const z = [];
    for (let i = 0; i < 20; i++) {
      const row = [];
      for (let j = 0; j < 20; j++) {
        row.push(Math.exp(-(Math.pow(i - 10, 2) + Math.pow(j - 10, 2)) / 50) * uncertaintyThreshold + Math.random() * 0.05);
      }
      z.push(row);
    }

    return {
      data: [{
        z: z,
        type: 'heatmap' as const,
        colorscale: 'RdBu',
        showscale: true,
        colorbar: { title: { text: 'Uncertainty σ' } }
      }],
      layout: {
        title: { text: 'Uncertainty Distribution Map', font: { size: 16, family: 'Roboto' } },
        xaxis: { title: 'Design Parameter 1' },
        yaxis: { title: 'Design Parameter 2' },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 50, r: 70, b: 50, l: 50 }
      },
      config: { responsive: true, displayModeBar: false }
    };
  };

  // Real-time Training Metrics
  const generateTrainingMetrics = () => {
    const epochs = Array.from({ length: 100 }, (_, i) => i + 1);
    const accuracy = epochs.map(e => 0.6 + 0.35 * (1 - Math.exp(-e / 20)) + Math.sin(e / 10) * 0.02);
    const loss = epochs.map(e => 2.5 * Math.exp(-e / 15) + Math.random() * 0.05);
    const uncertainty = epochs.map(e => 0.15 * Math.exp(-e / 25) + Math.random() * 0.005);

    return {
      data: [
        {
          x: epochs,
          y: accuracy,
          type: 'scatter' as const,
          mode: 'lines+markers' as const,
          name: 'Validation Accuracy',
          line: { color: '#2e7d32', width: 3 },
          marker: { size: 4 }
        },
        {
          x: epochs,
          y: loss,
          type: 'scatter' as const,
          mode: 'lines+markers' as const,
          name: 'Training Loss',
          yaxis: 'y2',
          line: { color: '#d32f2f', width: 3 },
          marker: { size: 4 }
        },
        {
          x: epochs,
          y: uncertainty,
          type: 'scatter' as const,
          mode: 'lines+markers' as const,
          name: 'Epistemic Uncertainty',
          yaxis: 'y3',
          line: { color: '#ed6c02', width: 3 },
          marker: { size: 4 }
        }
      ],
      layout: {
        title: { text: 'Real-time Training Metrics', font: { size: 16, family: 'Roboto' } },
        xaxis: { title: 'Training Epoch' },
        yaxis: { title: 'Accuracy', color: '#2e7d32' },
        yaxis2: { title: 'Loss', side: 'right', overlaying: 'y', color: '#d32f2f' },
        yaxis3: { title: 'Uncertainty', side: 'right', overlaying: 'y', position: 0.92, color: '#ed6c02' },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.8)' },
        margin: { t: 50, r: 120, b: 50, l: 50 }
      },
      config: { responsive: true, displayModeBar: false }
    };
  };

  // Attention Mechanism Visualization
  const generateAttentionMatrix = () => {
    const size = 12;
    const matrix = [];
    const features = ['Temperature', 'Pressure', 'Flow Rate', 'Material', 'Geometry', 'Boundary', 'Load', 'Time', 'Position X', 'Position Y', 'Velocity', 'Stress'];

    for (let i = 0; i < size; i++) {
      const row = [];
      for (let j = 0; j < size; j++) {
        // Create realistic attention patterns
        let attention = Math.exp(-Math.abs(i - j) / 2);
        if (i === j) attention = 1; // Self-attention
        if ((i < 4 && j < 4) || (i >= 4 && i < 8 && j >= 4 && j < 8)) attention *= 1.5; // Feature groups
        row.push(attention + Math.random() * 0.1);
      }
      matrix.push(row);
    }

    return {
      data: [{
        z: matrix,
        x: features,
        y: features,
        type: 'heatmap' as const,
        colorscale: 'Blues',
        showscale: true,
        colorbar: { title: { text: 'Attention Weight' } }
      }],
      layout: {
        title: { text: 'Multi-Head Attention Mechanism', font: { size: 16, family: 'Roboto' } },
        xaxis: { title: 'Input Features', tickangle: 45 },
        yaxis: { title: 'Output Features' },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 50, r: 70, b: 120, l: 100 }
      },
      config: { responsive: true, displayModeBar: false }
    };
  };

  // Bayesian Inference Distribution
  const generateBayesianDistribution = () => {
    const x = Array.from({ length: 100 }, (_, i) => -3 + i * 0.06);
    const prior = x.map(val => Math.exp(-Math.pow(val, 2) / 2) / Math.sqrt(2 * Math.PI));
    const posterior = x.map(val => Math.exp(-Math.pow(val-0.5, 2) / 0.8) / Math.sqrt(2 * Math.PI * 0.8));
    const likelihood = x.map(val => Math.exp(-Math.pow(val-0.3, 2) / 1.2) / Math.sqrt(2 * Math.PI * 1.2));

    return {
      data: [
        {
          x: x,
          y: prior,
          type: 'scatter' as const,
          mode: 'lines' as const,
          name: 'Prior',
          line: { color: '#1976d2', width: 3 },
          fill: 'tonexty' as const,
          fillcolor: 'rgba(25, 118, 210, 0.1)'
        },
        {
          x: x,
          y: likelihood,
          type: 'scatter' as const,
          mode: 'lines' as const,
          name: 'Likelihood',
          line: { color: '#ed6c02', width: 3 }
        },
        {
          x: x,
          y: posterior,
          type: 'scatter' as const,
          mode: 'lines' as const,
          name: 'Posterior',
          line: { color: '#2e7d32', width: 4 },
          fill: 'tonexty' as const,
          fillcolor: 'rgba(46, 125, 50, 0.2)'
        }
      ],
      layout: {
        title: { text: 'Bayesian Parameter Inference', font: { size: 16, family: 'Roboto' } },
        xaxis: { title: 'Parameter Value' },
        yaxis: { title: 'Probability Density' },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.8)' },
        margin: { t: 50, r: 20, b: 50, l: 50 }
      },
      config: { responsive: true, displayModeBar: false }
    };
  };

  // Graph Network Topology
  const generateGraphTopology = () => {
    const nodes = 30;
    const edges = [];
    const nodeX = [];
    const nodeY = [];
    const edgeX = [];
    const edgeY = [];

    // Generate circular layout
    for (let i = 0; i < nodes; i++) {
      const angle = (i / nodes) * 2 * Math.PI;
      nodeX.push(Math.cos(angle));
      nodeY.push(Math.sin(angle));
    }

    // Generate edges
    for (let i = 0; i < nodes; i++) {
      const connections = Math.floor(Math.random() * 3) + 1;
      for (let j = 0; j < connections; j++) {
        const target = Math.floor(Math.random() * nodes);
        if (target !== i) {
          edgeX.push(nodeX[i], nodeX[target], null);
          edgeY.push(nodeY[i], nodeY[target], null);
        }
      }
    }

    return {
      data: [
        {
          x: edgeX,
          y: edgeY,
          mode: 'lines' as const,
          line: { width: 1, color: '#888' },
          hoverinfo: 'none',
          showlegend: false
        },
        {
          x: nodeX,
          y: nodeY,
          mode: 'markers+text' as const,
          marker: {
            size: Array.from({ length: nodes }, () => Math.random() * 20 + 10),
            color: Array.from({ length: nodes }, () => Math.random()),
            colorscale: 'Viridis',
            showscale: true,
            colorbar: { title: { text: 'Node Feature' } }
          },
          text: Array.from({ length: nodes }, (_, i) => `N${i}`),
          textposition: 'middle center' as const,
          textfont: { size: 8, color: 'white' },
          hovertemplate: 'Node %{text}<br>Feature: %{marker.color:.2f}<extra></extra>',
          name: 'Graph Nodes'
        }
      ],
      layout: {
        title: { text: 'Graph Neural Network Topology', font: { size: 16, family: 'Roboto' } },
        showlegend: false,
        xaxis: { showgrid: false, zeroline: false, showticklabels: false },
        yaxis: { showgrid: false, zeroline: false, showticklabels: false },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 50, r: 70, b: 20, l: 20 }
      },
      config: { responsive: true, displayModeBar: false }
    };
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setSelectedTab(newValue);
  };

  return (
    <Box sx={{
      flexGrow: 1,
      p: 3,
      background: `linear-gradient(135deg, ${alpha(theme.palette.primary.light, 0.1)} 0%, ${alpha(theme.palette.secondary.light, 0.05)} 100%)`,
      minHeight: '100vh'
    }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" component="h1" sx={{
          fontWeight: 700,
          color: theme.palette.primary.main,
          mb: 1,
          display: 'flex',
          alignItems: 'center',
          gap: 2
        }}>
          <Analytics sx={{ fontSize: 40 }} />
          Advanced ML Visualizations
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ mb: 2 }}>
          Interactive Engineering Analysis & Neural Network Insights
        </Typography>

        {/* Control Panel */}
        <Paper elevation={2} sx={{ p: 2, mb: 3, bgcolor: alpha(theme.palette.background.paper, 0.9) }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Model Type</InputLabel>
                <Select
                  value={selectedModel}
                  label="Model Type"
                  onChange={(e) => setSelectedModel(e.target.value)}
                >
                  <MenuItem value="bayesian">Bayesian Neural Network</MenuItem>
                  <MenuItem value="graph">Graph Neural Network</MenuItem>
                  <MenuItem value="transformer">Transformer Model</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={3}>
              <Box sx={{ px: 2 }}>
                <Typography gutterBottom>Uncertainty Threshold</Typography>
                <Slider
                  value={uncertaintyThreshold}
                  onChange={(e, newValue) => setUncertaintyThreshold(newValue as number)}
                  min={0.01}
                  max={0.5}
                  step={0.01}
                  valueLabelDisplay="auto"
                  size="small"
                />
              </Box>
            </Grid>
            <Grid item xs={12} md={3}>
              <Box sx={{ px: 2 }}>
                <Typography gutterBottom>Mesh Resolution</Typography>
                <Slider
                  value={meshResolution}
                  onChange={(e, newValue) => setMeshResolution(newValue as number)}
                  min={10}
                  max={100}
                  step={10}
                  valueLabelDisplay="auto"
                  size="small"
                />
              </Box>
            </Grid>
            <Grid item xs={12} md={3}>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={realTimeMode}
                      onChange={(e) => setRealTimeMode(e.target.checked)}
                      size="small"
                    />
                  }
                  label="Real-time"
                />
                <Tooltip title="Toggle Controls">
                  <IconButton size="small" color="primary" onClick={handleToggleControls}>
                    <Settings />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Refresh Data">
                  <IconButton size="small" color="primary">
                    <Refresh />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Export Visualization">
                  <IconButton size="small" color="secondary" onClick={handleExportVisualization}>
                    <Download />
                  </IconButton>
                </Tooltip>
              </Box>
            </Grid>
          </Grid>
        </Paper>

        {/* Advanced Controls Panel */}
        {showControls && (
          <Paper elevation={2} sx={{ p: 2, mb: 3, bgcolor: alpha(theme.palette.secondary.light, 0.1) }}>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Timeline />
              Advanced Visualization Controls
            </Typography>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} md={2}>
                <Box sx={{ px: 1 }}>
                  <Typography gutterBottom variant="caption">Stress Scale</Typography>
                  <Slider
                    value={stressScale}
                    onChange={(e, newValue) => setStressScale(newValue as number)}
                    min={0.1}
                    max={3}
                    step={0.1}
                    valueLabelDisplay="auto"
                    size="small"
                  />
                </Box>
              </Grid>
              <Grid item xs={12} md={2}>
                <Box sx={{ px: 1 }}>
                  <Typography gutterBottom variant="caption">Contour Levels</Typography>
                  <Slider
                    value={contourLevels}
                    onChange={(e, newValue) => setContourLevels(newValue as number)}
                    min={5}
                    max={25}
                    step={1}
                    valueLabelDisplay="auto"
                    size="small"
                  />
                </Box>
              </Grid>
              <Grid item xs={12} md={2}>
                <Box sx={{ px: 1 }}>
                  <Typography gutterBottom variant="caption">Mesh Opacity</Typography>
                  <Slider
                    value={meshOpacity}
                    onChange={(e, newValue) => setMeshOpacity(newValue as number)}
                    min={0.1}
                    max={1}
                    step={0.1}
                    valueLabelDisplay="auto"
                    size="small"
                  />
                </Box>
              </Grid>
              <Grid item xs={12} md={2}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={showColorbar}
                      onChange={(e) => setShowColorbar(e.target.checked)}
                      size="small"
                    />
                  }
                  label="Show Colorbar"
                />
              </Grid>
              <Grid item xs={12} md={2}>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Tooltip title={isAnimating ? "Stop Animation" : "Start Animation"}>
                    <IconButton
                      size="small"
                      color={isAnimating ? "error" : "success"}
                      onClick={isAnimating ? handleStopAnimation : handleStartAnimation}
                    >
                      {isAnimating ? <Stop /> : <PlayArrow />}
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Reset View">
                    <IconButton size="small" onClick={handleResetView}>
                      <Refresh />
                    </IconButton>
                  </Tooltip>
                </Box>
              </Grid>
              <Grid item xs={12} md={2}>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                  <Chip label={`Scale: ${stressScale.toFixed(1)}`} size="small" />
                  <Chip label={`Levels: ${contourLevels}`} size="small" />
                </Box>
              </Grid>
            </Grid>
          </Paper>
        )}
      </Box>

      {/* Visualization Tabs */}
      <Paper elevation={3} sx={{ bgcolor: alpha(theme.palette.background.paper, 0.95) }}>
        <Tabs
          value={selectedTab}
          onChange={handleTabChange}
          variant="scrollable"
          scrollButtons="auto"
          sx={{
            borderBottom: 1,
            borderColor: 'divider',
            '& .MuiTab-root': {
              minHeight: 72,
              textTransform: 'none',
              fontSize: '1rem',
              fontWeight: 600
            }
          }}
        >
          <Tab
            icon={<BubbleChart />}
            label="3D Mesh Analysis"
            id="visualization-tab-0"
            aria-controls="visualization-tabpanel-0"
          />
          <Tab
            icon={<ShowChart />}
            label="Training Metrics"
            id="visualization-tab-1"
            aria-controls="visualization-tabpanel-1"
          />
          <Tab
            icon={<ScatterPlot />}
            label="Uncertainty Map"
            id="visualization-tab-2"
            aria-controls="visualization-tabpanel-2"
          />
          <Tab
            icon={<Memory />}
            label="Attention Matrix"
            id="visualization-tab-3"
            aria-controls="visualization-tabpanel-3"
          />
          <Tab
            icon={<TrendingUp />}
            label="Bayesian Inference"
            id="visualization-tab-4"
            aria-controls="visualization-tabpanel-4"
          />
          <Tab
            icon={<Timeline />}
            label="Graph Topology"
            id="visualization-tab-5"
            aria-controls="visualization-tabpanel-5"
          />
        </Tabs>

        <TabPanel value={selectedTab} index={0}>
          <Box sx={{
            p: 1,
            height: 600,
            backgroundColor: '#1e1e1e',
            border: '1px solid #3e3e42',
            borderTop: 'none'
          }}>
            <Plot
              data={generate3DMeshData().data}
              layout={generate3DMeshData().layout}
              config={generate3DMeshData().config}
              style={{ width: '100%', height: '100%' }}
            />
          </Box>
        </TabPanel>

        <TabPanel value={selectedTab} index={1}>
          <Box sx={{
            p: 1,
            height: 600,
            backgroundColor: '#1e1e1e',
            border: '1px solid #3e3e42',
            borderTop: 'none'
          }}>
            <Plot
              data={generateTrainingMetrics().data}
              // @ts-ignore - Plotly typing issue with yaxis2 overlaying property
              layout={generateTrainingMetrics().layout}
              config={generateTrainingMetrics().config}
              style={{ width: '100%', height: '100%' }}
            />
          </Box>
        </TabPanel>

        <TabPanel value={selectedTab} index={2}>
          <Box sx={{
            p: 1,
            height: 600,
            backgroundColor: '#1e1e1e',
            border: '1px solid #3e3e42',
            borderTop: 'none'
          }}>
            <Plot
              data={generateUncertaintyHeatmap().data}
              layout={generateUncertaintyHeatmap().layout}
              config={generateUncertaintyHeatmap().config}
              style={{ width: '100%', height: '100%' }}
            />
          </Box>
        </TabPanel>

        <TabPanel value={selectedTab} index={3}>
          <Box sx={{
            p: 1,
            height: 600,
            backgroundColor: '#1e1e1e',
            border: '1px solid #3e3e42',
            borderTop: 'none'
          }}>
            <Plot
              data={generateAttentionMatrix().data}
              layout={generateAttentionMatrix().layout}
              config={generateAttentionMatrix().config}
              style={{ width: '100%', height: '100%' }}
            />
          </Box>
        </TabPanel>

        <TabPanel value={selectedTab} index={4}>
          <Box sx={{
            p: 1,
            height: 600,
            backgroundColor: '#1e1e1e',
            border: '1px solid #3e3e42',
            borderTop: 'none'
          }}>
            <Plot
              data={generateBayesianDistribution().data}
              layout={generateBayesianDistribution().layout}
              config={generateBayesianDistribution().config}
              style={{ width: '100%', height: '100%' }}
            />
          </Box>
        </TabPanel>

        <TabPanel value={selectedTab} index={5}>
          <Box sx={{
            p: 1,
            height: 600,
            backgroundColor: '#1e1e1e',
            border: '1px solid #3e3e42',
            borderTop: 'none'
          }}>
            <Plot
              data={generateGraphTopology().data}
              layout={generateGraphTopology().layout}
              config={generateGraphTopology().config}
              style={{ width: '100%', height: '100%' }}
            />
          </Box>
        </TabPanel>
      </Paper>

      {/* Status Bar */}
      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Chip
            icon={<PlayArrow />}
            label={`Model: ${selectedModel}`}
            color="primary"
            variant="outlined"
          />
          <Chip
            icon={<TrendingUp />}
            label={`Uncertainty: ±${uncertaintyThreshold.toFixed(3)}`}
            color="secondary"
            variant="outlined"
          />
          <Chip
            icon={<BubbleChart />}
            label={`Resolution: ${meshResolution}x${meshResolution}`}
            color="success"
            variant="outlined"
          />
        </Box>
        <Typography variant="body2" color="text.secondary">
          {realTimeMode ? '🟢 Real-time monitoring active' : '⏸️ Static view'}
        </Typography>
      </Box>
    </Box>
  );
}