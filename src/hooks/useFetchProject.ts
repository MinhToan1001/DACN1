import { useEffect, useState } from 'react';
import { fetchProject } from '../api/projectApi';
import { Project } from '../types/Project';

export const useFetchProject = () => {
  const [project, setProject] = useState<Project[]>([]);
  const [loadding, setLoading] = useState(true);

  useEffect(() => {
    fetchProject()
      .then(setProject)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  return { project, loadding };
};
