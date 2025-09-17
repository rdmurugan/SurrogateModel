import React, { createContext, useContext, useState, useEffect } from 'react';

export interface Project {
  id: string;
  name: string;
  description: string;
  type: 'structural' | 'thermal' | 'fluid' | 'multiphysics';
  created: string;
  modified: string;
  author: string;
  meshData?: any;
  materials?: any[];
  boundaryConditions?: any[];
  results?: any[];
  settings?: any;
}

interface ProjectContextType {
  currentProject: Project | null;
  projects: Project[];
  createProject: (project: Omit<Project, 'id' | 'created' | 'modified'>) => void;
  openProject: (projectId: string) => void;
  saveProject: (project: Project) => void;
  deleteProject: (projectId: string) => void;
  exportProject: (projectId: string, format: string) => void;
  importProject: (file: File) => Promise<void>;
}

const ProjectContext = createContext<ProjectContextType | undefined>(undefined);

export function ProjectProvider({ children }: { children: React.ReactNode }) {
  const [currentProject, setCurrentProject] = useState<Project | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);

  useEffect(() => {
    // Load projects from localStorage
    const savedProjects = localStorage.getItem('cae_projects');
    if (savedProjects) {
      try {
        setProjects(JSON.parse(savedProjects));
      } catch (error) {
        console.error('Failed to load projects:', error);
      }
    }

    // Load current project
    const currentProjectId = localStorage.getItem('current_project_id');
    if (currentProjectId && savedProjects) {
      const parsedProjects = JSON.parse(savedProjects);
      const project = parsedProjects.find((p: Project) => p.id === currentProjectId);
      if (project) {
        setCurrentProject(project);
      }
    }
  }, []);

  const saveProjectsToStorage = (projectsList: Project[]) => {
    localStorage.setItem('cae_projects', JSON.stringify(projectsList));
    setProjects(projectsList);
  };

  const createProject = (projectData: Omit<Project, 'id' | 'created' | 'modified'>) => {
    const newProject: Project = {
      ...projectData,
      id: 'proj_' + Date.now(),
      created: new Date().toISOString(),
      modified: new Date().toISOString(),
    };

    const updatedProjects = [...projects, newProject];
    saveProjectsToStorage(updatedProjects);
    setCurrentProject(newProject);
    localStorage.setItem('current_project_id', newProject.id);
  };

  const openProject = (projectId: string) => {
    const project = projects.find(p => p.id === projectId);
    if (project) {
      setCurrentProject(project);
      localStorage.setItem('current_project_id', projectId);
    }
  };

  const saveProject = (project: Project) => {
    const updatedProject = {
      ...project,
      modified: new Date().toISOString(),
    };

    const updatedProjects = projects.map(p =>
      p.id === project.id ? updatedProject : p
    );

    saveProjectsToStorage(updatedProjects);
    setCurrentProject(updatedProject);
  };

  const deleteProject = (projectId: string) => {
    const updatedProjects = projects.filter(p => p.id !== projectId);
    saveProjectsToStorage(updatedProjects);

    if (currentProject?.id === projectId) {
      setCurrentProject(null);
      localStorage.removeItem('current_project_id');
    }
  };

  const exportProject = (projectId: string, format: string) => {
    const project = projects.find(p => p.id === projectId);
    if (!project) return;

    let content: string;
    let filename: string;
    let mimeType: string;

    switch (format) {
      case 'json':
        content = JSON.stringify(project, null, 2);
        filename = `${project.name}.json`;
        mimeType = 'application/json';
        break;
      case 'inp':
        // Abaqus INP format
        content = generateInpFile(project);
        filename = `${project.name}.inp`;
        mimeType = 'text/plain';
        break;
      case 'nas':
        // Nastran format
        content = generateNasFile(project);
        filename = `${project.name}.nas`;
        mimeType = 'text/plain';
        break;
      default:
        return;
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const importProject = async (file: File) => {
    const text = await file.text();

    try {
      if (file.name.endsWith('.json')) {
        const projectData = JSON.parse(text);
        const importedProject: Project = {
          ...projectData,
          id: 'proj_' + Date.now(),
          created: new Date().toISOString(),
          modified: new Date().toISOString(),
          name: `${projectData.name} (Imported)`,
        };

        const updatedProjects = [...projects, importedProject];
        saveProjectsToStorage(updatedProjects);
        setCurrentProject(importedProject);
        localStorage.setItem('current_project_id', importedProject.id);
      } else {
        // Parse other CAE formats (INP, NAS, etc.)
        const parsedProject = parseCAEFile(file.name, text);
        createProject(parsedProject);
      }
    } catch (error) {
      throw new Error('Failed to import project: ' + error);
    }
  };

  const value = {
    currentProject,
    projects,
    createProject,
    openProject,
    saveProject,
    deleteProject,
    exportProject,
    importProject,
  };

  return <ProjectContext.Provider value={value}>{children}</ProjectContext.Provider>;
}

export function useProject() {
  const context = useContext(ProjectContext);
  if (context === undefined) {
    throw new Error('useProject must be used within a ProjectProvider');
  }
  return context;
}

// Helper functions for file format conversion
function generateInpFile(project: Project): string {
  let content = `*HEADING\n${project.name} - Generated from CAE Platform\n\n`;

  // Add nodes
  if (project.meshData?.nodes) {
    content += '*NODE\n';
    project.meshData.nodes.forEach((node: any) => {
      content += `${node.id}, ${node.x}, ${node.y}, ${node.z}\n`;
    });
    content += '\n';
  }

  // Add elements
  if (project.meshData?.elements) {
    content += '*ELEMENT, TYPE=C3D4\n';
    project.meshData.elements.forEach((element: any) => {
      content += `${element.id}, ${element.nodeIds.join(', ')}\n`;
    });
    content += '\n';
  }

  // Add materials
  if (project.materials) {
    project.materials.forEach((material: any) => {
      content += `*MATERIAL, NAME=${material.name}\n`;
      content += `*ELASTIC\n`;
      content += `${material.properties.youngsModulus}, ${material.properties.poissonsRatio}\n`;
      content += `*DENSITY\n`;
      content += `${material.properties.density}\n\n`;
    });
  }

  return content;
}

function generateNasFile(project: Project): string {
  let content = `$NASTRAN INPUT FILE\n$${project.name}\n\n`;

  // Add nodes
  if (project.meshData?.nodes) {
    project.meshData.nodes.forEach((node: any) => {
      content += `GRID    ${node.id.toString().padStart(8)}        ${node.x.toFixed(6).padStart(8)}${node.y.toFixed(6).padStart(8)}${node.z.toFixed(6).padStart(8)}\n`;
    });
    content += '\n';
  }

  // Add elements
  if (project.meshData?.elements) {
    project.meshData.elements.forEach((element: any) => {
      content += `CTETRA  ${element.id.toString().padStart(8)}       1${element.nodeIds.map((id: number) => id.toString().padStart(8)).join('')}\n`;
    });
    content += '\n';
  }

  return content;
}

function parseCAEFile(filename: string, content: string): Omit<Project, 'id' | 'created' | 'modified'> {
  const baseName = filename.split('.')[0];

  return {
    name: baseName,
    description: `Imported from ${filename}`,
    type: 'structural',
    author: 'Imported',
    meshData: {
      nodes: [],
      elements: []
    },
    materials: [],
    boundaryConditions: [],
    results: []
  };
}