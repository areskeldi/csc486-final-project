Yryskeldi Emilbek uulu  
Dr. Patrick Shepherd  
Berea College
20 April 2022
# CSC 486: Network Dynamics: Final Project

Project Report: [Click here]()

Network nodes: [Click here](https://github.com/areskeldi/csc486-finalproject/blob/main/Last.fm-dataset.zip).      

Network edges: [Click here]()

Model code: [Click here]()

Project Prototype Milestone:  

As part of my project, I have wrote code that conducted a modification to a conventional Independent Cascade diffusion algorithm. Under this modification, the initial seed process will include not only the selected seed nodes (using algorithm 'a') but also self-activated nodes (“self-activation occurs in many real-world situations; for example, people naturally share product recommendations with their friends, even without marketing intervention”). I have also shifted from using a Flixster dataset for my project (which contained over 9 mln edges) to using a Last.fm dataset. Both datasets represent similar entities, where nodes reflect users and edges reflect the friendships between users.  

I successfully modified the Independent Cascade diffusion model to include self-activated nodes on top of the nodes selected by the algorithm 'a'. Now, I need to producing several visualizations to explore the improvements that this modification results in. I would like to run several statistical tests, to see if there's a positive correlation between the number of initial seed nodes and the size of the influence set within the same graph or if there is a statistically significant difference in the maximum size of the influence set between small-world and scale-free networks? I also want to learn how to read the 'csv' files to initiate the nodes and edges using the 'networkx' library.
