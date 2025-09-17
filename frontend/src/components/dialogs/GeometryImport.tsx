import React, { useState, useRef } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  Card,
  CardContent,
  LinearProgress,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  IconButton,
  Chip,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Divider
} from '@mui/material';
import {
  CloudUpload,
  InsertDriveFile,
  CheckCircle,
  Warning,
  Error,
  Delete,
  Visibility,
  VisibilityOff,
  Engineering,
  Architecture,
  Category
} from '@mui/icons-material';

interface GeometryFile {
  id: string;
  name: string;
  type: 'step' | 'iges' | 'stl' | 'obj' | 'parasolid' | 'catia' | 'solidworks';
  size: number;
  status: 'pending' | 'importing' | 'success' | 'error';
  progress?: number;
  error?: string;
  units: 'mm' | 'cm' | 'm' | 'in' | 'ft';
  bodies: number;
  surfaces: number;
  visible: boolean;
}

interface GeometryImportProps {
  open: boolean;
  onClose: () => void;
  onImport?: (files: GeometryFile[]) => void;
}

const GeometryImport: React.FC<GeometryImportProps> = ({ open, onClose, onImport }) => {
  const [files, setFiles] = useState<GeometryFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [importSettings, setImportSettings] = useState({
    units: 'mm' as const,
    healGeometry: true,
    mergeCoplanarFaces: false,
    removeSmallFeatures: true,
    createAssembly: true
  });
  const fileInputRef = useRef<HTMLInputElement>(null);

  const supportedFormats = [
    { ext: 'step', name: 'STEP Files', mime: '.step,.stp' },
    { ext: 'iges', name: 'IGES Files', mime: '.iges,.igs' },
    { ext: 'stl', name: 'STL Files', mime: '.stl' },
    { ext: 'obj', name: 'OBJ Files', mime: '.obj' },
    { ext: 'x_t', name: 'Parasolid Files', mime: '.x_t,.x_b' }
  ];

  const getFileIcon = (type: string) => {
    switch (type) {
      case 'step':
      case 'iges':
        return <Engineering color="primary" />;
      case 'stl':
      case 'obj':
        return <Architecture color="secondary" />;
      default:
        return <InsertDriveFile color="action" />;
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success':
        return <CheckCircle color="success" />;
      case 'error':
        return <Error color="error" />;
      case 'importing':
        return null; // Will show progress bar
      default:
        return <Warning color="warning" />;
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const handleFileSelect = (selectedFiles: FileList | null) => {
    if (!selectedFiles) return;

    const newFiles: GeometryFile[] = Array.from(selectedFiles).map((file, index) => {
      const extension = file.name.split('.').pop()?.toLowerCase() || 'unknown';
      const fileType = ['step', 'stp'].includes(extension) ? 'step' :
                      ['iges', 'igs'].includes(extension) ? 'iges' :
                      extension === 'stl' ? 'stl' :
                      extension === 'obj' ? 'obj' : 'step';

      return {
        id: `file_${Date.now()}_${index}`,
        name: file.name,
        type: fileType as GeometryFile['type'],
        size: file.size,
        status: 'pending' as const,
        units: importSettings.units,
        bodies: Math.floor(Math.random() * 5) + 1,
        surfaces: Math.floor(Math.random() * 50) + 10,
        visible: true
      };
    });

    setFiles(prev => [...prev, ...newFiles]);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFileSelect(e.dataTransfer.files);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleFileInputClick = () => {
    fileInputRef.current?.click();
  };

  const handleImportFiles = () => {
    // Simulate import process
    const filesToImport = files.filter(f => f.status === 'pending');

    filesToImport.forEach((file, index) => {
      setTimeout(() => {
        setFiles(prev => prev.map(f =>
          f.id === file.id ? { ...f, status: 'importing' as const, progress: 0 } : f
        ));

        // Simulate progress
        const progressInterval = setInterval(() => {
          setFiles(prev => prev.map(f => {
            if (f.id === file.id && f.status === 'importing') {
              const newProgress = (f.progress || 0) + Math.random() * 15 + 5;
              if (newProgress >= 100) {
                clearInterval(progressInterval);
                return {
                  ...f,
                  status: Math.random() > 0.8 ? 'error' as const : 'success' as const,
                  progress: 100,
                  error: Math.random() > 0.8 ? 'Invalid geometry format' : undefined
                };
              }
              return { ...f, progress: newProgress };
            }
            return f;
          }));
        }, 200);
      }, index * 500);
    });
  };

  const handleRemoveFile = (fileId: string) => {
    setFiles(prev => prev.filter(f => f.id !== fileId));
  };

  const handleToggleVisibility = (fileId: string) => {
    setFiles(prev => prev.map(f =>
      f.id === fileId ? { ...f, visible: !f.visible } : f
    ));
  };

  const handleFinishImport = () => {
    const successfulFiles = files.filter(f => f.status === 'success');
    if (onImport) {
      onImport(successfulFiles);
    }
    onClose();
  };

  const canImport = files.some(f => f.status === 'pending');
  const hasSuccessful = files.some(f => f.status === 'success');
  const isImporting = files.some(f => f.status === 'importing');

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Typography variant="h6" component="div" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <CloudUpload />
          Import Geometry
        </Typography>
      </DialogTitle>
      <DialogContent>
        <Box sx={{ mb: 3 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Import CAD geometry files to use in your structural analysis
          </Typography>

          {/* Supported Formats */}
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>Supported Formats:</Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {supportedFormats.map((format) => (
                <Chip
                  key={format.ext}
                  label={format.ext.toUpperCase()}
                  size="small"
                  variant="outlined"
                />
              ))}
            </Box>
          </Box>

          {/* Import Settings */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>Import Settings</Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Units</InputLabel>
                    <Select
                      value={importSettings.units}
                      label="Units"
                      onChange={(e) => setImportSettings({ ...importSettings, units: e.target.value as any })}
                    >
                      <MenuItem value="mm">Millimeters</MenuItem>
                      <MenuItem value="cm">Centimeters</MenuItem>
                      <MenuItem value="m">Meters</MenuItem>
                      <MenuItem value="in">Inches</MenuItem>
                      <MenuItem value="ft">Feet</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={importSettings.healGeometry}
                          onChange={(e) => setImportSettings({ ...importSettings, healGeometry: e.target.checked })}
                        />
                      }
                      label={<Typography variant="body2">Heal Geometry</Typography>}
                    />
                    <FormControlLabel
                      control={
                        <Switch
                          checked={importSettings.createAssembly}
                          onChange={(e) => setImportSettings({ ...importSettings, createAssembly: e.target.checked })}
                        />
                      }
                      label={<Typography variant="body2">Create Assembly</Typography>}
                    />
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {/* File Drop Zone */}
          <Card
            sx={{
              border: isDragging ? '2px dashed #1976d2' : '2px dashed #ccc',
              backgroundColor: isDragging ? 'rgba(25, 118, 210, 0.05)' : 'transparent',
              cursor: 'pointer',
              mb: 3
            }}
            onClick={handleFileInputClick}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            <CardContent sx={{ textAlign: 'center', py: 4 }}>
              <CloudUpload sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                Drop files here or click to browse
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Select CAD files to import into your project
              </Typography>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".step,.stp,.iges,.igs,.stl,.obj,.x_t,.x_b"
                style={{ display: 'none' }}
                onChange={(e) => handleFileSelect(e.target.files)}
              />
            </CardContent>
          </Card>

          {/* File List */}
          {files.length > 0 && (
            <Card>
              <CardContent>
                <Typography variant="subtitle1" gutterBottom>
                  Files to Import ({files.length})
                </Typography>
                <List dense>
                  {files.map((file) => (
                    <ListItem
                      key={file.id}
                      sx={{
                        border: '1px solid #e0e0e0',
                        borderRadius: 1,
                        mb: 1,
                        backgroundColor: file.visible ? 'transparent' : 'rgba(0, 0, 0, 0.05)'
                      }}
                    >
                      <ListItemIcon>
                        {getFileIcon(file.type)}
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="body2">{file.name}</Typography>
                            <Chip label={file.type.toUpperCase()} size="small" />
                            <Typography variant="caption" color="text.secondary">
                              {formatFileSize(file.size)}
                            </Typography>
                          </Box>
                        }
                        secondary={
                          <Box sx={{ mt: 1 }}>
                            {file.status === 'importing' && file.progress !== undefined ? (
                              <Box>
                                <LinearProgress variant="determinate" value={file.progress} />
                                <Typography variant="caption">Importing... {Math.round(file.progress)}%</Typography>
                              </Box>
                            ) : (
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                {getStatusIcon(file.status)}
                                <Typography variant="caption">
                                  {file.status === 'success' && `${file.bodies} bodies, ${file.surfaces} surfaces`}
                                  {file.status === 'error' && file.error}
                                  {file.status === 'pending' && 'Ready to import'}
                                </Typography>
                              </Box>
                            )}
                          </Box>
                        }
                      />
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <IconButton
                          size="small"
                          onClick={() => handleToggleVisibility(file.id)}
                          color={file.visible ? 'primary' : 'default'}
                        >
                          {file.visible ? <Visibility /> : <VisibilityOff />}
                        </IconButton>
                        <IconButton
                          size="small"
                          onClick={() => handleRemoveFile(file.id)}
                          color="error"
                        >
                          <Delete />
                        </IconButton>
                      </Box>
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          )}

          {/* Status Summary */}
          {files.length > 0 && (
            <Box sx={{ mt: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <Chip
                label={`Pending: ${files.filter(f => f.status === 'pending').length}`}
                size="small"
                color="default"
              />
              <Chip
                label={`Importing: ${files.filter(f => f.status === 'importing').length}`}
                size="small"
                color="info"
              />
              <Chip
                label={`Success: ${files.filter(f => f.status === 'success').length}`}
                size="small"
                color="success"
              />
              <Chip
                label={`Errors: ${files.filter(f => f.status === 'error').length}`}
                size="small"
                color="error"
              />
            </Box>
          )}
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        {canImport && (
          <Button
            variant="contained"
            onClick={handleImportFiles}
            disabled={isImporting}
            startIcon={<CloudUpload />}
          >
            {isImporting ? 'Importing...' : 'Import Files'}
          </Button>
        )}
        {hasSuccessful && !isImporting && (
          <Button
            variant="contained"
            color="success"
            onClick={handleFinishImport}
          >
            Finish Import
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default GeometryImport;