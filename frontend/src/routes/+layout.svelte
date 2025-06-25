<script lang="ts">
	import '../app.css';

	let { children } = $props();
</script>

<div class="">
		<nav class="flex items-center bg-gray-200 px-4 py-3 justify-between shadow-xl">
			<div class="flex flex-col">
				<span class="text-lg font-bold text	">Machine Learning Web Aplication</span>
				<span class="text-sm text	">PUC Minas • Ciências de Dados</span>
			</div>
			<a class="ml-4 rounded-md shadow-2xl bg-gray-300 hover:bg-yellow-500 px-4 py-2 font-semibold text-black duration-300 transition-colors hover:" href="https://github.com/caioceo/machineLearning-webapp" target="_blank" rel="noopener noreferrer">GitHub</a>
		</nav>
	</div>

<div class="container mx-auto">
	{@render children()}
</div>
